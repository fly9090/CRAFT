import os
import math
import torch
import argparse
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from accelerate import Accelerator
from transformers import AdamW, get_linear_schedule_with_warmup
from Data_prepare_s import make_data_for_gpt_pretrain
from tqdm import tqdm
import warnings
import gc
import logging

warnings.filterwarnings("ignore")


# ==========================================
# 1. 辅助函数
# ==========================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_grad_norm(model):
    """计算模型所有参数梯度的L2范数"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequence_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask.to(seq.device)


# ==========================================
# 2. 模型组件 (已参数化)
# ==========================================

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, maxlen, d_model):
        super(EmbeddingLayer, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(maxlen, d_model)
        self.atchley_embed = nn.Linear(5, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(0.3)

    def forward(self, x, atchley_inputs=None):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long).to(x.device)
        pos = pos.unsqueeze(0).expand_as(x)
        embedding = self.tok_embed(x) + self.pos_embed(pos)
        if atchley_inputs is not None:
            embedding += self.atchley_embed(atchley_inputs)
        return self.drop(self.norm(embedding))


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.drop = nn.Dropout(0.3)

    def forward(self, Q, K, V, attn_mask):
        # Q: [batch, heads, len, d_k]
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = self.drop(nn.Softmax(dim=-1)(scores))
        context = torch.matmul(attn, V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(0.3)

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)
        # Linear projections & split heads
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # Scaled Dot Product Attention
        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)

        # Concat heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.dense(context)
        return self.norm(self.drop(output) + Q)  # Residual


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.ac = nn.GELU()
        self.drop = nn.Dropout(0.3)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.fc2(self.ac(self.fc1(x)))
        x = self.drop(x)
        return self.norm(x + residual)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, dec_inputs, dec_self_attn_mask):
        dec_outputs = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs


class GPT_Model(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layers=12, n_heads=12, d_ff=3072, maxlen=32):
        super(GPT_Model, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size, maxlen, d_model)

        # 使用参数化创建 Layer
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])

        self.cls = nn.Linear(d_model, vocab_size)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ 初始化权重 (Normal 0.02) """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, atchley_inputs=None):
        decoder_out = self.embedding(input_ids, atchley_inputs=atchley_inputs)
        dec_self_attn_pad_mask = get_attn_pad_mask(input_ids, input_ids)
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(input_ids)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0)

        for layer in self.layers:
            decoder_out = layer(decoder_out, dec_self_attn_mask)

        pre = self.cls(decoder_out)
        return decoder_out, pre


# ==========================================
# 3. 训练流程
# ==========================================

def create_parser():
    parser = argparse.ArgumentParser(description="GPT pretrain",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", dest="data_path", type=str, required=True)
    parser.add_argument("--vocab_path", dest="vocab_path", type=str, required=True)
    parser.add_argument("--model_path", dest="model_path", type=str, required=True)
    parser.add_argument("--epoch", dest="epoch", type=int, default=10, required=False)
    parser.add_argument("--maxlen", dest="maxlen", type=int, default=80, required=False)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=64, required=False)
    parser.add_argument("--gradient_accumulation_steps", dest="gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=3e-4, required=False)
    parser.add_argument("--random_seed", type=int, dest="random_seed", default=0, required=False)
    args = parser.parse_args()
    return args


def main(args):
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    set_seed(args.random_seed)

    # 日志配置
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
        print(f"日志文件路径: {log_file}")
        logger.info("=" * 50)
        logger.info(f"Start Training. Args: {args}")
        print("开始加载数据...")

    train_dataset, validate_dataset, vocab_size = make_data_for_gpt_pretrain(args.data_path, args.vocab_path,
                                                                             args.maxlen)

    if accelerator.is_local_main_process:
        print(f"数据加载完成，vocab_size: {vocab_size}")
        logger.info(f"Data Loaded. Vocab Size: {vocab_size}")

    data_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validate_dataloader = Data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False)

    # 这里不需要传参了，因为我们已经恢复了默认值。当然，也可以显式传参。
    model = GPT_Model(vocab_size, maxlen=args.maxlen)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    num_update_steps_per_epoch = math.ceil(len(data_loader) / args.gradient_accumulation_steps)
    total_steps = num_update_steps_per_epoch * args.epoch
    warmup_steps = 0.1 * total_steps

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    model, optimizer, data_loader, validate_dataloader, scheduler = accelerator.prepare(
        model, optimizer, data_loader, validate_dataloader, scheduler
    )

    if accelerator.is_local_main_process:
        print(f"开始训练... 物理 Batch: {args.batch_size}, 累积步数: {args.gradient_accumulation_steps}")

    for epoch in range(args.epoch):
        model.train()
        total_loss = 0

        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1} Train", disable=not accelerator.is_local_main_process)

        for input_ids, output_labels, mask_for_loss, atchley_inputs in pbar:
            with accelerator.accumulate(model):
                _, logits_lm = model(input_ids, atchley_inputs=atchley_inputs)

                logits_view = logits_lm.view(-1, vocab_size)
                labels_view = output_labels.view(-1)

                loss = criterion(logits_view, labels_view)

                accelerator.backward(loss)

                grad_norm = 0.0
                if accelerator.sync_gradients:
                    grad_norm = get_grad_norm(model)
                    if grad_norm > 10.0 and accelerator.is_local_main_process:
                        msg = f"High Gradient detected: {grad_norm:.4f}"
                        print(f"\n[Warning] {msg}")
                        logger.warning(msg)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                del logits_lm, logits_view, labels_view, loss

            if accelerator.sync_gradients and accelerator.is_local_main_process:
                pbar.set_postfix({"Loss": f"{total_loss / (pbar.n + 1):.3f}", "Grad": f"{grad_norm:.3f}"})

            if torch.cuda.is_available() and torch.cuda.memory_reserved() > 0.85 * torch.cuda.get_device_properties(0).total_memory:
                gc.collect()
                torch.cuda.empty_cache()

        avg_train_loss = total_loss / len(data_loader)
        accelerator.print(f"epoch:{epoch + 1}, done! Avg Train Loss: {avg_train_loss}")

        model.eval()
        loss_v = 0
        with torch.no_grad():
            for input_ids, output_labels, mask_for_loss, atchley_inputs in tqdm(validate_dataloader,
                                                                                desc=f"Epoch {epoch + 1} Val",
                                                                                disable=not accelerator.is_local_main_process):
                _, logits_lm = model(input_ids, atchley_inputs=atchley_inputs)
                logits_view = logits_lm.view(-1, vocab_size)
                labels_view = output_labels.view(-1)
                loss = criterion(logits_view, labels_view)
                loss_v += loss.item()
                del logits_lm, logits_view, labels_view, loss

        gc.collect()
        torch.cuda.empty_cache()

        val_len = len(validate_dataloader)
        avg_val_loss = loss_v / val_len if val_len > 0 else 0.0
        accelerator.print(f"loss in validate_dataset: {avg_val_loss}")

        if accelerator.is_local_main_process:
            logger.info(
                f"Epoch [{epoch + 1}/{args.epoch}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Last Grad: {grad_norm:.4f}")

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(unwrapped_model.state_dict(), args.model_path)


if __name__ == "__main__":
    args = create_parser()
    main(args)