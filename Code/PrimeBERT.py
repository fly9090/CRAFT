import math
import re
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import argparse
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from transformers import AdamW, get_linear_schedule_with_warmup
from Data_prepare_s import make_data_for_pretrain
from tqdm import tqdm
import warnings
import random
import logging
import gc

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
    """计算模型梯度的L2范数"""
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


# ==========================================
# 2. 模型组件
# ==========================================

class Embedding(nn.Module):
    def __init__(self, vocab_size, maxlen, d_model, dropout=0.1):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(maxlen, d_model)
        self.atchley_embed = nn.Linear(5, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.maxlen = maxlen

    def forward(self, x, atchley_inputs=None):
        seq_len = x.size(1)
        if seq_len > self.maxlen:
            x = x[:, :self.maxlen]
            if atchley_inputs is not None:
                atchley_inputs = atchley_inputs[:, :self.maxlen, :]
            seq_len = self.maxlen

        pos = torch.arange(seq_len, dtype=torch.long).to(x.device)
        pos = pos.unsqueeze(0).expand_as(x)

        embedding = self.tok_embed(x) + self.pos_embed(pos)
        if atchley_inputs is not None:
            embedding += self.atchley_embed(atchley_inputs)

        return self.drop(self.norm(embedding))


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.drop = nn.Dropout(dropout)

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(64)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        attn = self.drop(attn)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=768, n_heads=12, d_k=64, d_v=64, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.fc = nn.Linear(n_heads * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.attn_func = ScaledDotProductAttention(dropout=dropout)

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

        return self.norm(self.drop(output) + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model=768, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return self.norm(self.drop(output) + residual)


class EncoderLayer(nn.Module):
    def __init__(self, d_model=768, n_heads=12, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, dropout=dropout)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class BERT(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layers=12, maxlen=80, dropout=0.3):
        super(BERT, self).__init__()
        self.embedding = Embedding(vocab_size, maxlen, d_model, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, 12, dropout) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, d_model)
        self.activ = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, vocab_size)

        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, masked_pos, atchley_inputs=None):
        output = self.embedding(input_ids, atchley_inputs=atchley_inputs)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)

        attns = []
        for layer in self.layers:
            output, attn = layer(output, enc_self_attn_mask)
            attns.append(attn)

        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))
        h_masked = torch.gather(output, 1, masked_pos)
        h_masked = self.norm(self.activ(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias

        return output, logits_lm


# ==========================================
# 3. 兼容加载函数 (Smart Fault Tolerance)
# ==========================================
def load_compatible_weights(model, weight_path, device):
    if not os.path.exists(weight_path):
        print(f"No checkpoint found at {weight_path}, starting from scratch.")
        return
    try:
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"\n[Warning] Failed to load checkpoint: {e}")
        print("[Action] File seems corrupted. Starting from scratch.\n")
        return

    current_model_dict = model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        if k in current_model_dict:
            target_shape = current_model_dict[k].shape
            if v.shape != target_shape:
                print(f"Resizing {k}: {v.shape} -> {target_shape}")
                if len(target_shape) == 2:
                    min_0 = min(target_shape[0], v.shape[0])
                    min_1 = min(target_shape[1], v.shape[1])
                    new_param = torch.zeros(target_shape, device=v.device)
                    new_param[:min_0, :min_1] = v[:min_0, :min_1]
                    v = new_param
                elif len(target_shape) == 1:
                    min_0 = min(target_shape[0], v.shape[0])
                    new_param = torch.zeros(target_shape, device=v.device)
                    new_param[:min_0] = v[:min_0]
                    v = new_param
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    print(f"Successfully loaded weights from {weight_path}")


# ==========================================
# 4. 主流程
# ==========================================

def create_parser():
    parser = argparse.ArgumentParser(description="bert_pretrain6",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", dest="data_path", type=str, required=True)
    parser.add_argument("--vocab_path", dest="vocab_path", type=str, required=True)
    parser.add_argument("--model_path", dest="model_path", type=str, required=True)
    parser.add_argument("--epoch", dest="epoch", type=int, default=10)
    parser.add_argument("--maxlen", dest="maxlen", type=int, default=80)
    parser.add_argument("--max_pred", dest="max_pred", type=int, default=12)#80*0.15
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=3e-4)
    parser.add_argument("--random_seed", type=int, default=0)

    # [修改点] 增强正则化参数默认值 (方案二)
    parser.add_argument("--dropout", type=float, default=0.3, help="Increased dropout for stronger regularization")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="Increased weight decay for stronger regularization")

    return parser.parse_args()


def main(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs]
    )
    set_seed(args.random_seed)

    logger = None
    if accelerator.is_local_main_process:
        log_file = args.model_path.rsplit('.', 1)[0] + '.log'
        logging.basicConfig(
            filename=log_file, filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, force=True
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Start Training BERT. Args: {args}")
        print(f"日志文件路径: {log_file}")

    train_dataset, validate_dataset, vocab_size = make_data_for_pretrain(
        args.data_path, args.vocab_path, args.maxlen, args.max_pred
    )

    data_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validate_dataloader = Data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False)

    model = BERT(vocab_size, maxlen=args.maxlen, dropout=args.dropout)

    device_map = torch.device('cpu')
    load_compatible_weights(model, args.model_path, device_map)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    num_update_steps_per_epoch = math.ceil(len(data_loader) / args.gradient_accumulation_steps)
    total_steps = num_update_steps_per_epoch * args.epoch

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps),
                                                num_training_steps=total_steps)

    model, optimizer, data_loader, validate_dataloader, scheduler = accelerator.prepare(
        model, optimizer, data_loader, validate_dataloader, scheduler
    )

    if accelerator.is_local_main_process:
        print(f"开始训练... 物理 Batch: {args.batch_size}, 累积步数: {args.gradient_accumulation_steps}")

    best_val_loss = float('inf')

    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        last_grad_norm = 0.0

        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1} Train", disable=not accelerator.is_local_main_process)

        for input_ids, masked_tokens, masked_pos, atchley_inputs in pbar:
            with accelerator.accumulate(model):
                _, logits_lm = model(input_ids, masked_pos, atchley_inputs=atchley_inputs)
                loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)
                accelerator.backward(loss_lm)

                if accelerator.sync_gradients:
                    last_grad_norm = get_grad_norm(model)
                    if last_grad_norm > 10.0 and accelerator.is_local_main_process:
                        logger.warning(f"High Gradient: {last_grad_norm:.4f}")

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss_lm.item()
                del logits_lm, loss_lm

            if accelerator.is_local_main_process:
                pbar.set_postfix({
                    "Loss": f"{total_loss / (pbar.n + 1):.3f}",
                    "Grad": f"{last_grad_norm:.3f}"
                })

            if torch.cuda.is_available() and torch.cuda.memory_reserved() > 0.85 * torch.cuda.get_device_properties(0).total_memory:
                gc.collect()
                torch.cuda.empty_cache()

        avg_train_loss = total_loss / len(data_loader)
        accelerator.print(f"epoch:{epoch + 1}, done! Avg Train Loss: {avg_train_loss}")

        # Validation Loop
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for input_ids, masked_tokens, masked_pos, atchley_inputs in tqdm(validate_dataloader,
                                                                             desc=f"Epoch {epoch + 1} Val",
                                                                             disable=not accelerator.is_local_main_process):
                _, logits_lm = model(input_ids, masked_pos, atchley_inputs=atchley_inputs)
                loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)
                total_val_loss += loss_lm.item()
                del logits_lm, loss_lm

        accelerator.wait_for_everyone()  # 同步

        avg_val_loss = total_val_loss / len(validate_dataloader) if len(validate_dataloader) > 0 else 0.0
        accelerator.print(f"Epoch {epoch + 1} Val Loss: {avg_val_loss:.4f}")

        if accelerator.is_local_main_process:
            logger.info(
                f"Epoch [{epoch + 1}/{args.epoch}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                accelerator.print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
                logger.info(f"New best model saved at epoch {epoch + 1}")
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(), args.model_path)
            else:
                accelerator.print(f"Validation loss did not improve (Best: {best_val_loss:.4f}).")


if __name__ == "__main__":
    args = create_parser()
    main(args)