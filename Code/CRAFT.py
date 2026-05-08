import os
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import argparse
import logging
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from transformers import AdamW, get_linear_schedule_with_warmup

# === 导入自定义模块 ===
from BERT_s import BERT, set_seed
from GPT_s import GPT_Model, get_attn_pad_mask, get_attn_subsequence_mask
from Data_prepare_s import make_data_for_seq2seq
from beam_search_s import expand_inputs

from utility_s import (
    compute_struct_constraint,
    compute_novelty,
    compute_affinity_score,
    get_atchley_factors,
    _avg_atchley_for_token
)

# ==========================================
# 1. 辅助类与函数
# ==========================================

ATCHLEY_TENSOR = None


def get_atchley_tensor_global(vocab_size, idx2token, device):
    matrix = torch.zeros(vocab_size, 5).to(device)
    for i in range(vocab_size):
        token = idx2token.get(i, '[PAD]')
        vec = _avg_atchley_for_token(token)
        matrix[i] = torch.tensor(vec, device=device)
    return matrix


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(0.3)

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(64)
        scores.masked_fill_(attn_mask, -1e9)
        attn = self.drop(nn.Softmax(dim=-1)(scores))
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=768, n_heads=12, d_k=64, d_v=64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.norm = nn.LayerNorm(self.d_model)
        self.attn_func = ScaledDotProductAttention()

    def forward(self, input_Q, input_K, input_V, attn_mask):
        batch_size = input_Q.size(0)
        residual = input_Q
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = self.attn_func(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        return self.norm(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=False),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model, bias=False)
        )
        self.norm = nn.LayerNorm(self.d_model)
        self.drop = nn.Dropout(0.3)

    def forward(self, inputs):
        residual = inputs
        output = self.drop(self.fc(inputs))
        return self.norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self, n_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs, bert_enc_outputs):
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            bert_enc_outputs, enc_self_attn = layer(bert_enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return bert_enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self, n_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs, gpt_dec_outputs):
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            gpt_dec_outputs, dec_self_attn, dec_enc_attn = layer(gpt_dec_outputs, enc_outputs, dec_self_attn_mask,
                                                                 dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return gpt_dec_outputs, dec_self_attns, dec_enc_attns


class CRA(nn.Module):
    def __init__(self, bert, gpt, vocabsize, d_model=768):
        super().__init__()
        self.bert = bert
        self.gpt = gpt
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, vocabsize, bias=False)
        self.d_model = d_model

    def forward(self, enc_inputs, dec_inputs, masked_pos, enc_atchley=None, dec_atchley=None):
        enc_outputs_bert, _ = self.bert(enc_inputs, masked_pos, atchley_inputs=enc_atchley)
        enc_outputs, enc_self_attns = self.encoder(enc_inputs, enc_outputs_bert)

        gpt_output = self.gpt(dec_inputs, atchley_inputs=dec_atchley)

        dec_outputs_gpt = None
        if isinstance(gpt_output, tuple):
            val1, val2 = gpt_output
            if val1.size(-1) == self.d_model:
                dec_outputs_gpt = val1
            elif val2.size(-1) == self.d_model:
                dec_outputs_gpt = val2
            else:
                raise RuntimeError(
                    f"GPT output shapes mismatch d_model ({self.d_model}). Got: {val1.shape}, {val2.shape}")
        else:
            dec_outputs_gpt = gpt_output

        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs, dec_outputs_gpt)
        dec_outputs = dec_outputs + dec_outputs_gpt
        dec_logits = self.projection(dec_outputs)

        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns


def compute_accuracy_token(logits, labels, mask):
    pred_tokens = torch.argmax(logits, dim=-1)
    correct = (pred_tokens == labels) & (mask == 1)
    return correct.sum().float(), mask.sum().float()


def load_compatible_weights(model, weight_path, device):
    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    new_state_dict = {}
    current_model_dict = model.state_dict()

    for k, v in state_dict.items():
        new_k = k
        if 'enc_self_attn' in k and 'dec_self_attn' not in k:
            new_k = k.replace('enc_self_attn', 'dec_self_attn')
        if 'layer_norm1' in k:
            new_k = new_k.replace('layer_norm1', 'dec_self_attn.norm')
        elif 'layer_norm2' in k:
            new_k = new_k.replace('layer_norm2', 'pos_ffn.norm')
        elif 'fc.weight' in k and 'dec_self_attn' in new_k:
            new_k = new_k.replace('fc', 'dense')
        elif 'fc.bias' in k and 'dec_self_attn' in new_k:
            new_k = new_k.replace('fc', 'dense')
        if 'atchley_proj' in k:
            new_k = k.replace('atchley_proj', 'atchley_embed')

        if new_k in current_model_dict:
            target_shape = current_model_dict[new_k].shape
            if v.shape != target_shape:
                print(f"Shape mismatch for {new_k}: checkpoint {v.shape} != current {target_shape}. Resizing...")
                if len(target_shape) == 2 and target_shape[1] == v.shape[1]:
                    min_vocab = min(target_shape[0], v.shape[0])
                    new_param = torch.zeros(target_shape, device=v.device)
                    new_param[:min_vocab, :] = v[:min_vocab, :]
                    v = new_param
                elif len(target_shape) == 1 and target_shape[0] != v.shape[0]:
                    min_vocab = min(target_shape[0], v.shape[0])
                    new_param = torch.zeros(target_shape, device=v.device)
                    new_param[:min_vocab] = v[:min_vocab]
                    v = new_param
                else:
                    continue
        new_state_dict[new_k] = v

    try:
        model.load_state_dict(new_state_dict, strict=False)
        print(f"Successfully loaded weights from {weight_path} with mapping and resizing fixes.")
    except Exception as e:
        print(f"Error loading weights: {e}")


def create_parser():
    parser = argparse.ArgumentParser(description="CRA seq2seq", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", dest="data_path", type=str, required=True, help="Path to MIRA/Training CSV")
    parser.add_argument("--tcr_vocab_path", dest="tcr_vocab_path", type=str, required=True)
    parser.add_argument("--pep_vocab_path", dest="pep_vocab_path", type=str, required=True)
    parser.add_argument("--model_path", dest="model_path", type=str, required=True, help="Path to save/load CRA model")
    parser.add_argument("--bert_path", dest="bert_path", type=str, required=True, help="Path to pretrained BERT")
    parser.add_argument("--gpt_path", dest="gpt_path", type=str, required=True, help="Path to pretrained GPT")
    parser.add_argument("--result_path", dest="result_path", type=str, default="results.csv")
    parser.add_argument("--mode", dest="mode", type=str, required=True, help="'train' or 'generate'")
    parser.add_argument("--epoch", dest="epoch", type=int, default=10)

    parser.add_argument("--beam", dest="beam", type=int, default=10, help="Number of samples per epitope")
    parser.add_argument("--topk", dest="topk", type=int, default=20, help="Top-K sampling (High Diversity)")
    parser.add_argument("--temperature", dest="temperature", type=float, default=0.5,
                        help="Sampling temperature (lower=more conservative/higher quality)")
    parser.add_argument("--top_k_per_epitope", type=int, default=5,
                        help="After generation, keep top-K beams per epitope ranked by log-prob score (0=keep all)")
    parser.add_argument("--length_penalty", type=float, default=0.7,
                        help="Length penalty exponent for beam scoring (higher=favor shorter sequences)")

    # [关键修复] 将 maxlen 默认值从 32 改为 80！
    # 32 太短了，连 MHC 34 个氨基酸都装不下，更别说后面的 Epitope 了。
    parser.add_argument("--maxlen", dest="maxlen", type=int, default=80)

    parser.add_argument("--batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=5e-5)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--constraint_lambda", type=float, default=0.5, help="Weight for structural constraint loss")
    parser.add_argument("--mhc_allele", type=str, default='HLA-A*02:01',
                        help="当 e_pmhc.csv 中查找不到某抗原肽时，使用该 HLA 等位基因名称作为回退 MHC（需配合 --hla_csv_path）")
    # ---- 动态 MHC 匹配（新增参数）----
    parser.add_argument("--e_pmhc_path", type=str,
                        default='/data/CRAFT/Data/MIRA/e_pmhc.csv',
                        help="e_pmhc.csv 路径，包含 epitope→pmhc pseudo sequence 映射；"
                             "提供后将按每条样本的抗原肽序列动态查找对应 MHC，同一抗原肽多个 MHC 时自动扩增样本。")
    parser.add_argument("--hla_csv_path", type=str,
                        default='/data/CRAFT/Data/MIRA/hla_pseudo_seqs.csv',
                        help="hla_pseudo_seqs.csv 路径，包含 allele→pseudo-sequence 映射；"
                             "用于将 --mhc_allele 名称解析为 pseudo sequence（回退场景）。")
    return parser.parse_args()


def main(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    set_seed(args.random_seed)

    logger = None
    if accelerator.is_local_main_process:
        log_file = args.model_path.rsplit('.', 1)[0] + '.log'
        logging.basicConfig(
            filename=log_file,
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            force=True
        )
        logger = logging.getLogger(__name__)
        print(f"日志将保存至: {log_file}")
        logger.info("=" * 50)
        logger.info(f"Start CRA Task. Mode: {args.mode}")
        logger.info(f"Args: {args}")

    if args.mode == 'train':
        accelerator.print("正在准备训练数据 (Mode: Train)...")
        # 传入动态 MHC 映射文件路径，实现逐样本 MHC 匹配
        dataset, validate_dataset, vocab_size_tcr, vocab_size_pep, tcr_idx2token, pep_idx2token = make_data_for_seq2seq(
            args.data_path,
            args.tcr_vocab_path,
            args.pep_vocab_path,
            mhc_allele=args.mhc_allele,
            mode=args.mode,
            max_len=args.maxlen,
            e_pmhc_path=args.e_pmhc_path,
            hla_csv_path=args.hla_csv_path,
        )
        data_loader = Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=8,  # 关键！开启多进程加载
    pin_memory=True) # 关键！加速传输
        validate_dataloader = Data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False)

        bert = BERT(vocab_size_pep)
        gpt = GPT_Model(vocab_size_tcr, d_model=768, n_layers=12, n_heads=12, d_ff=3072, maxlen=args.maxlen)
        device_map = torch.device('cpu')
        try:
            bert.load_state_dict(torch.load(args.bert_path, map_location=device_map, weights_only=True))
        except Exception:
            accelerator.print("Warning: BERT strict loading failed, trying non-strict.")
            bert.load_state_dict(torch.load(args.bert_path, map_location=device_map, weights_only=True), strict=False)
        load_compatible_weights(gpt, args.gpt_path, device_map)

        for name, param in bert.named_parameters():
            if 'linear' in name or 'fc' in name: param.requires_grad = False

        model = CRA(bert, gpt, vocab_size_tcr)
        optimizer = AdamW([{"params": [p for p in model.parameters() if p.requires_grad]}], lr=args.learning_rate)
        total_steps = len(data_loader) * args.epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps),
                                                    num_training_steps=total_steps)
        criterion = nn.CrossEntropyLoss(reduction='none')

        model, optimizer, data_loader, validate_dataloader, scheduler = accelerator.prepare(
            model, optimizer, data_loader, validate_dataloader, scheduler
        )

        global ATCHLEY_TENSOR
        ATCHLEY_TENSOR = get_atchley_tensor_global(vocab_size_tcr, tcr_idx2token, accelerator.device)

        accelerator.print(f"开始训练，共 {args.epoch} 轮...")

        for epoch in range(args.epoch):
            model.train()
            total_loss_ep = 0.0
            total_acc_num = 0.0
            total_acc_denom = 0.0
            pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1} Train", disable=not accelerator.is_local_main_process)
            for batch in pbar:
                optimizer.zero_grad()
                dec_in, out_lbl, mask, enc_in, enc_att, dec_att, epi_att = batch
                masked_pos = torch.zeros((dec_in.size(0), 5), dtype=torch.long).to(dec_in.device)
                logits, _, _, _ = model(enc_in, dec_in, masked_pos, enc_atchley=enc_att, dec_atchley=dec_att)

                ce = criterion(logits.transpose(1, 2), out_lbl)

                # 所有样本均为正样本，直接用 mask 计算语言模型损失
                loss_lm = (ce * mask).sum() / mask.sum().clamp_min(1.0)

                # ── 约束损失（结构对比）──────────────────────────────────────────
                # dec_mean：模型预测的 TCR Atchley 均值（可微，梯度流经 logits）
                probs        = F.softmax(logits, dim=-1)
                soft_atchley = torch.matmul(probs, ATCHLEY_TENSOR)
                dec_mean = (soft_atchley * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0)

                # enc_mean：仅抗原肽（epitope-only）的 Atchley 均值
                # epi_att 中非零位置即为真实 epitope token（PAD 位全零向量）
                epi_mask = (epi_att.abs().sum(-1) > 0).float()          # [B, max_len]
                enc_mean = (epi_att * epi_mask.unsqueeze(-1)).sum(1) / epi_mask.sum(1, keepdim=True).clamp_min(1.0)

                # Triplet hinge loss：正例相似度 > 负例（打乱 epitope）相似度 + margin
                cos_sim_pos  = F.cosine_similarity(dec_mean, enc_mean, dim=1)
                idx          = torch.randperm(dec_in.size(0), device=dec_in.device)
                enc_mean_neg = enc_mean[idx]
                cos_sim_neg  = F.cosine_similarity(dec_mean, enc_mean_neg, dim=1)
                loss_cons    = args.constraint_lambda * F.relu(cos_sim_neg - cos_sim_pos + 0.1).mean()
                # ────────────────────────────────────────────────────────────────

                total_loss = loss_lm + loss_cons

                acc_n, acc_d = compute_accuracy_token(logits, out_lbl, mask)
                total_acc_num += acc_n.item()
                total_acc_denom += acc_d.item()
                total_loss_ep += total_loss.item()

                accelerator.backward(total_loss)
                optimizer.step()
                scheduler.step()
                curr_acc = total_acc_num / total_acc_denom if total_acc_denom > 0 else 0
                pbar.set_postfix(
                    {"Loss": f"{loss_lm.item():.3f}", "Cons": f"{loss_cons.item():.3f}", "Acc": f"{curr_acc:.3f}"})

            # 验证：计算平均验证 Loss（不再依赖正/负标签，无需 AUC）
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(validate_dataloader, desc=f"Epoch {epoch + 1} Val",
                                  disable=not accelerator.is_local_main_process):
                    dec_in, out_lbl, mask, enc_in, enc_att, dec_att, epi_att = batch
                    masked_pos = torch.zeros((dec_in.size(0), 5), dtype=torch.long).to(dec_in.device)
                    logits, _, _, _ = model(enc_in, dec_in, masked_pos, enc_atchley=enc_att, dec_atchley=dec_att)
                    eval_mask = (out_lbl != 0).float()
                    ce = criterion(logits.transpose(1, 2), out_lbl)
                    loss_per_sample = (ce * eval_mask).sum(1) / eval_mask.sum(1).clamp_min(1.0)
                    total_val_loss += loss_per_sample.mean().item()

            avg_val_loss = total_val_loss / len(validate_dataloader) if len(validate_dataloader) > 0 else 0.0
            avg_train_loss = total_loss_ep / len(data_loader)
            accelerator.print(
                f"Epoch {epoch + 1} Done. Mean Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            if accelerator.is_local_main_process:
                logger.info(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

            accelerator.save(model.state_dict(), args.model_path)

    elif args.mode == 'generate':
        accelerator.print("正在准备生成数据 (Mode: Generate)...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 传入动态 MHC 映射文件路径，生成时同样按抗原肽匹配对应 MHC
        dataset, tcr_idx2token, pep_idx2token, vocab_size_tcr, vocab_size_pep = make_data_for_seq2seq(
            args.data_path,
            args.tcr_vocab_path,
            args.pep_vocab_path,
            mhc_allele=args.mhc_allele,
            mode=args.mode,
            max_len=args.maxlen,
            e_pmhc_path=args.e_pmhc_path,
            hla_csv_path=args.hla_csv_path,
        )
        data_loader = Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        bert = BERT(vocab_size_pep)
        gpt = GPT_Model(vocab_size_tcr, d_model=768, n_layers=12, n_heads=12, d_ff=3072, maxlen=args.maxlen)
        model = CRA(bert, gpt, vocab_size_tcr)

        state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)

        model.to(device)
        model.eval()

        train_df = pd.read_csv(args.data_path)
        if 'beta' in train_df.columns:
            train_seqs_set = set(train_df['beta'].astype(str).tolist())
        else:
            print("Warning: 'beta' column not found in data CSV; novelty check will mark all sequences as novel.")
            train_seqs_set = set()

        results_list = []

        print(f"开始生成 (Top-K Sampling, K={args.topk}, Temp={args.temperature})...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader)):
                inputIds, enc_atchley = batch
                inputIds = inputIds.to(device)
                enc_atchley = enc_atchley.to(device)
                batchSize = inputIds.shape[0]

                masked_pos = torch.zeros((batchSize, 1), dtype=torch.long).to(device)
                inputEncodes, _ = model.bert(inputIds, masked_pos, atchley_inputs=enc_atchley)
                inputEncodes, _ = model.encoder(inputIds, inputEncodes)

                input_expanded_ids, input_expanded_encodes = expand_inputs(inputIds, inputEncodes, beam_size=args.beam)
                decoder_output_ids = torch.full((batchSize * args.beam, 1), 1, dtype=torch.long).to(device)

                # Track cumulative log-probability per beam for quality-based selection
                beam_log_probs = torch.zeros(batchSize * args.beam, device=device)

                cur_len = 1
                while cur_len < args.maxlen:
                    dec_ids_cpu = decoder_output_ids.cpu().tolist()
                    dec_atchley_list = []
                    for seq_ids in dec_ids_cpu:
                        toks = [tcr_idx2token.get(t, '[PAD]') for t in seq_ids]
                        vecs = [_avg_atchley_for_token(t) for t in toks]
                        dec_atchley_list.append(vecs)

                    dec_atchley = torch.tensor(dec_atchley_list, dtype=torch.float).to(device)

                    dec_outputs_gpt = model.gpt(decoder_output_ids, atchley_inputs=dec_atchley)
                    if isinstance(dec_outputs_gpt, tuple): dec_outputs_gpt = dec_outputs_gpt[0]

                    dec_outputs, _, _ = model.decoder(decoder_output_ids, input_expanded_ids, input_expanded_encodes,
                                                      dec_outputs_gpt)
                    dec_outputs = dec_outputs + dec_outputs_gpt
                    dec_logit = model.projection(dec_outputs)

                    next_token_logits = dec_logit[:, -1, :]
                    if args.topk > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, args.topk)[0][
                            ..., -1, None]
                        next_token_logits[indices_to_remove] = -float('Inf')

                    probs = F.softmax(next_token_logits / args.temperature, dim=-1)
                    next_token_ids = torch.multinomial(probs, num_samples=1)

                    # Accumulate log-probability of the selected token for beam scoring
                    # Use raw logits (before topk mask) for a proper model confidence score
                    raw_log_probs = F.log_softmax(dec_logit[:, -1, :] / args.temperature, dim=-1)
                    selected_log_probs = raw_log_probs.gather(1, next_token_ids).squeeze(-1)
                    beam_log_probs = beam_log_probs + selected_log_probs

                    decoder_output_ids = torch.cat([decoder_output_ids, next_token_ids], dim=-1)
                    cur_len += 1

                for i in range(batchSize * args.beam):
                    batch_orig_idx = i // args.beam

                    epi_seq_ids = inputIds[batch_orig_idx].cpu().tolist()
                    epi_str_full = ""
                    for x in epi_seq_ids:
                        if x in [0, 1, 2, 3]: continue
                        epi_str_full += pep_idx2token.get(x, '')

                    # if batch_idx == 0 and i == 0:
                    #     print(f"\n[Debug] Full Decoded Input: {epi_str_full}")

                    # [修复] 动态检测 MHC 长度
                    # 如果 input 确实被截断了（比如用户忘了改 maxlen），这里会保留原样
                    # 如果 maxlen 够长 (80)，这里会正确切掉 MHC (34 chars)
                    # mhc_pseudo_len = 34  # HLA-A*02:01 长度
                    # if len(epi_str_full) > mhc_pseudo_len:
                    #     epi_str = epi_str_full[mhc_pseudo_len:]
                    # else:
                    #     epi_str = epi_str_full  # Fallback
                    #
                    # if not epi_str:
                    #     epi_str = epi_str_full

                    if len(epi_str_full) > 34:
                        epi_str = epi_str_full[34:]
                    else:
                        print(f"Warning: Sequence too short to contain MHC+Epitope: {epi_str_full}")
                        epi_str = epi_str_full
                    if i == 0:
                        print(f"\n[检查] 原始序列: {epi_str_full}")
                        print(f"[检查] 切分结果: {epi_str}")
                        print("-" * 30)


                    tcr_seq_ids = decoder_output_ids[i].cpu().tolist()
                    tcr_str = ""
                    for x in tcr_seq_ids:
                        if x == 2: break
                        if x in [0, 1, 3]: continue
                        tcr_str += tcr_idx2token.get(x, '')

                    cons = compute_struct_constraint(tcr_str, epi_str)
                    aff = compute_affinity_score(tcr_str, epi_str)
                    is_novel = 1 if tcr_str not in train_seqs_set else 0
                    seq_len = max(len(tcr_str), 1)
                    norm_beam_score = beam_log_probs[i].item() / (seq_len ** args.length_penalty)

                    results_list.append({
                        'epitope': epi_str,
                        'generated_beta': tcr_str,
                        'constraint_score': cons,
                        'affinity_score': aff,
                        'is_novel': is_novel,
                        'beam_score': beam_log_probs[i].item(),
                        'norm_beam_score': norm_beam_score,
                        'pLDDT': 'Requires AlphaFold',
                        'scRMSD': 'Requires AlphaFold'
                    })

        df_res = pd.DataFrame(results_list)

        # Quality-based beam selection: keep top-K sequences per epitope ranked by
        # length-normalized log-probability score. This filters out low-confidence
        # samples that drag down BLEU4 and BLOSUM62 without improving diversity.
        if len(df_res) > 0 and args.top_k_per_epitope > 0:
            df_res = (df_res
                      .sort_values('norm_beam_score', ascending=False)
                      .groupby('epitope', sort=False)
                      .head(args.top_k_per_epitope)
                      .reset_index(drop=True))
            print(f"[质量筛选] top_k_per_epitope={args.top_k_per_epitope} → 保留 {len(df_res)} 条结果")

        if len(df_res) > 0:
            div = len(df_res['generated_beta'].unique()) / len(df_res)
            nov = df_res['is_novel'].mean()
            print(f"生成完成! 结果保存在: {args.result_path}")
            print(f"整体指标 -> Diversity: {div:.4f}, Novelty Rate: {nov:.4f}")
            if accelerator.is_local_main_process:
                logger.info(f"Generation Done. Result Path: {args.result_path}")
                logger.info(f"Metrics -> Diversity: {div:.4f}, Novelty Rate: {nov:.4f}")
        else:
            print("生成结果为空，请检查输入。")
            if accelerator.is_local_main_process:
                logger.warning("Generation produced empty results.")

        df_res.to_csv(args.result_path, index=False)


if __name__ == "__main__":
    args = create_parser()
    main(args)