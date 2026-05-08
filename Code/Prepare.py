import os
import torch
import random
import pandas as pd
import numpy as np
import torch.utils.data as Data

# 从 utility_s 导入辅助函数
from utility_s import _avg_atchley_for_token, build_atchley_for_ids


def get_mhc_pseudo(allele, hla_csv_path=None):
    """
    根据 HLA allele 名称返回对应的 34 位 pseudo sequence。
    优先从 hla_csv_path 所指向的 hla_pseudo_seqs.csv 中查找；
    文件不存在或查找失败时，回退到内置小字典。

    参数
    ----
    allele       : str  e.g. 'HLA-A*02:01'
    hla_csv_path : str | None  hla_pseudo_seqs.csv 的绝对路径
    """
    if hla_csv_path and os.path.isfile(hla_csv_path):
        try:
            hla_df = pd.read_csv(hla_csv_path)
            hla_df.columns = [c.strip() for c in hla_df.columns]
            matched = hla_df[hla_df['allele'] == allele]
            if not matched.empty:
                return str(matched['pseudo-sequence'].iloc[0]).strip()
        except Exception as e:
            print(f"[Warning] 读取 hla_pseudo_seqs.csv 失败: {e}，回退到内置字典。")

    # 内置回退字典（覆盖常见等位基因）
    pseudo_dict = {
        'HLA-A*02:01': 'YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY',
        'HLA-A*01:01': 'YSAMYEEKVAHTDENIAYLMFHYYTWAVQAYTGY',
        'HLA-A*03:01': 'YFAMYQENMAHTDANTLYIIYRDYTWVARVYRGY',
        'HLA-A*11:01': 'YFAMYQENVAHTDENTLYIIYRDYTWAELAYTWY',
        'HLA-A*24:02': 'YFAMYGEKVAQTDVNTLYIRYRYYTWAVLTYTWY',
        'HLA-B*07:02': 'YHAMYREKVAQTDVNTLYIRYRDYTWAELAYTWY',
        'HLA-B*08:01': 'YSAMYEEKVAHTDENIAYLMFHYYTWAVLAYEWY',
    }
    return pseudo_dict.get(allele, '')


def load_epitope_pmhc_map(e_pmhc_path):
    """
    读取 e_pmhc.csv，构建 epitope → [pmhc_seq, ...] 的映射字典。
    同一抗原肽可能被多个 MHC 等位基因呈递，因此值为列表。

    参数
    ----
    e_pmhc_path : str | None  e_pmhc.csv 的绝对路径

    返回
    ----
    dict[str, list[str]]  {epitope_seq: [mhc_pseudo_seq, ...]}
    空字典表示文件不可用或格式不正确。
    """
    if not e_pmhc_path or not os.path.isfile(e_pmhc_path):
        print(f"[Warning] e_pmhc.csv 文件未找到: '{e_pmhc_path}'，MHC 动态匹配将被跳过。")
        return {}
    try:
        df = pd.read_csv(e_pmhc_path)
        df.columns = [c.strip() for c in df.columns]
        if 'epitope' not in df.columns or 'pmhc' not in df.columns:
            print(f"[Warning] {e_pmhc_path} 缺少 'epitope' 或 'pmhc' 列，动态 MHC 匹配将被跳过。")
            return {}
        epi_pmhc_map = {}
        for _, row in df.iterrows():
            # 统一：去首尾空白 + 转大写，确保与查询侧保持一致
            epi  = str(row['epitope']).strip().upper()
            pmhc = str(row['pmhc']).strip().upper()
            epi_pmhc_map.setdefault(epi, [])
            if pmhc not in epi_pmhc_map[epi]:   # 去重
                epi_pmhc_map[epi].append(pmhc)
        hit = len(epi_pmhc_map)
        # 诊断输出：打印前 3 个键，便于与训练集抗原肽核对
        sample_keys = list(epi_pmhc_map.keys())[:3]
        print(f"[Data] 已从 {e_pmhc_path} 加载 {hit} 种抗原肽的 MHC pseudo 映射。")
        print(f"[Data] 字典示例键（前3条）: {sample_keys}")
        return epi_pmhc_map
    except Exception as e:
        print(f"[Warning] 读取 e_pmhc.csv 失败: {e}，动态 MHC 匹配将被跳过。")
        return {}


def tokenize_with_vocab_optimized(seq: str, token2idx, vocab_freq):
    if pd.isna(seq): return []
    seq = str(seq)
    seq = seq[:30] if len(seq) > 30 else seq
    tokens = []
    i, n = 0, len(seq)
    token_lens = [3, 2]

    while i < n:
        matched = False
        for length in token_lens:
            if i + length <= n:
                sub_seq = seq[i: i + length]
                if vocab_freq.get(sub_seq, 0) > 0:
                    tokens.append(token2idx.get(sub_seq, token2idx.get('[UNK]', 0)))
                    i += length
                    matched = True
                    break
        if not matched:
            char_token = seq[i]
            tokens.append(token2idx.get(char_token, token2idx.get('[UNK]', 0)))
            i += 1
    return tokens


# ==========================================
# Dataset 类
# ==========================================

class Dataset_for_epitope(Data.Dataset):
    def __init__(self, input_ids, masked_tokens, masked_pos, atchley_inputs):
        self.input_ids = input_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
        self.atchley_inputs = atchley_inputs

    def __len__(self): return len(self.input_ids)

    def __getitem__(self, idx): return self.input_ids[idx], self.masked_tokens[idx], self.masked_pos[idx], \
    self.atchley_inputs[idx]


class Dataset_for_beta(Data.Dataset):
    def __init__(self, input_ids, output_labels, mask_for_loss, atchley_inputs):
        self.input_ids = torch.LongTensor(input_ids)
        self.output_labels = torch.LongTensor(output_labels)
        self.mask_for_loss = torch.LongTensor(mask_for_loss)
        self.atchley_inputs = torch.FloatTensor(atchley_inputs)

    def __len__(self): return len(self.input_ids)

    def __getitem__(self, idx): return self.input_ids[idx], self.output_labels[idx], self.mask_for_loss[idx], \
    self.atchley_inputs[idx]


class Dataset_for_seq2seq(Data.Dataset):
    def __init__(self, dec_input_ids, output_labels, mask_for_loss,
                 enc_input_ids, enc_atchley, dec_atchley, epi_atchley):
        self.dec_input_ids = dec_input_ids
        self.output_labels = output_labels
        self.mask_for_loss = mask_for_loss
        self.enc_input_ids = enc_input_ids
        self.enc_atchley   = enc_atchley
        self.dec_atchley   = dec_atchley
        # 仅抗原肽（epitope-only）的 Atchley 特征，用于约束损失的目标表示
        self.epi_atchley   = epi_atchley

    def __len__(self):
        return len(self.dec_input_ids)

    def __getitem__(self, idx):
        return (self.dec_input_ids[idx], self.output_labels[idx], self.mask_for_loss[idx],
                self.enc_input_ids[idx], self.enc_atchley[idx], self.dec_atchley[idx],
                self.epi_atchley[idx])


class Dataset_for_generate(Data.Dataset):
    def __init__(self, inp, att):
        self.inp, self.att = inp, att

    def __len__(self): return len(self.inp)

    def __getitem__(self, i): return self.inp[i], self.att[i]


# ==========================================
# 数据处理
# ==========================================

def make_data_for_pretrain(data_path, vocab_path, maxlen, max_pred):
    print("[Data] Processing BERT Pretrain Data...")
    data_df = pd.read_csv(data_path)
    token_df = pd.read_csv(vocab_path)
    vocab_freq = {t: f for t, f in zip(token_df['token'], token_df['frequency'])}
    token_list = list(token_df['token']) + list("RHKDESTNQCUGPAVILMFYW")
    token2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3, '[UNK]': 4}
    start = len(token2idx)
    for i, w in enumerate(token_list):
        if w not in token2idx: token2idx[w] = i + start
    idx2token = {v: k for k, v in token2idx.items()}
    vocab_size = len(token2idx)

    batch = []
    for seq in data_df['epitope']:
        seq = str(seq)
        token_idx = tokenize_with_vocab_optimized(seq, token2idx, vocab_freq)
        input_ids = [token2idx['[CLS]']] + token_idx + [token2idx['[SEP]']]
        n_pad = maxlen - len(input_ids)
        if n_pad > 0:
            input_ids += [0] * n_pad

        # 创建 masked_pos 和 masked_tokens（随机掩码逻辑，从初始代码推断）
        n_pred = min(max_pred, max(1, int(len(token_idx) * 0.15)))
        cand_maked_pos = [i+1 for i in range(len(token_idx)) if token_idx[i] != token2idx['[UNK]']]  # 排除CLS/SEP
        random.shuffle(cand_maked_pos)
        masked_pos = cand_maked_pos[:n_pred]
        masked_tokens = [input_ids[pos] for pos in masked_pos]
        for pos in masked_pos:
            if random.random() < 0.8:  # 80% mask
                input_ids[pos] = token2idx['[MASK]']
            elif random.random() < 0.5:  # 10% random
                input_ids[pos] = random.choice(list(token2idx.values())[5:])  # 从vocab随机
        masked_pos += [0] * (max_pred - len(masked_pos))  # pad to max_pred
        masked_tokens += [0] * (max_pred - len(masked_tokens))

        atchley_inputs = build_atchley_for_ids(input_ids, idx2token, maxlen)
        batch.append([input_ids, masked_tokens, masked_pos, atchley_inputs])

    # 修改：替换 shuffle_and_split_data 为内置 random.shuffle 和手动 split
    random.shuffle(batch)  # shuffle batch list
    val_size = int(len(batch) * 0.1)  # ratio=0.1 for val
    validate_batch = batch[:val_size]
    train_batch = batch[val_size:]

    # 构建 input_ids 等列表
    train_input_ids, train_masked_tokens, train_masked_pos, train_atchley_inputs = zip(*train_batch)
    validate_input_ids, validate_masked_tokens, validate_masked_pos, validate_atchley_inputs = zip(*validate_batch)

    train_dataset = Dataset_for_epitope(
        torch.LongTensor(train_input_ids), torch.LongTensor(train_masked_tokens),
        torch.LongTensor(train_masked_pos), torch.FloatTensor(train_atchley_inputs)
    )
    validate_dataset = Dataset_for_epitope(
        torch.LongTensor(validate_input_ids), torch.LongTensor(validate_masked_tokens),
        torch.LongTensor(validate_masked_pos), torch.FloatTensor(validate_atchley_inputs)
    )

    return train_dataset, validate_dataset, vocab_size


def make_data_for_gpt_pretrain(data_path, vocab_path, max_len):
    data_df = pd.read_csv(data_path)
    seq_list = [str(seq) for seq in data_df['beta']]
    token_df = pd.read_csv(vocab_path)
    token_list = list(token_df['token']) + list("RHKDESTNQCUGPAVILMFYW")
    token2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3, '[UNK]': 4}
    start_idx = len(token2idx)
    for i, w in enumerate(token_list):
        if w not in token2idx: token2idx[w] = i + start_idx
    idx2token = {v: k for k, v in token2idx.items()}
    vocab_size = len(token2idx)
    vocab_freq_dict = {t: f for t, f in zip(token_df['token'], token_df['frequency'])}

    input_ids_list, output_labels_list, mask_for_loss_list, atchley_list = [], [], [], []

    print("[Data] Processing GPT Pretrain Data...")
    for seq in seq_list:
        tokens = tokenize_with_vocab_optimized(seq, token2idx, vocab_freq_dict)
        if len(tokens) < 2: continue
        input_id = tokens[:-1]
        output_label = tokens[1:]
        n_pad = max_len - len(input_id)
        if n_pad > 0:
            input_id.extend([0] * n_pad)
            output_label.extend([0] * n_pad)
            mask_for_loss = [1] * len(tokens[:-1]) + [0] * n_pad
        else:
            input_id = input_id[:max_len]
            output_label = output_label[:max_len]
            mask_for_loss = [1] * max_len

        atchley_seq = build_atchley_for_ids(input_id, idx2token, max_len)
        input_ids_list.append(input_id)
        output_labels_list.append(output_label)
        mask_for_loss_list.append(mask_for_loss)
        atchley_list.append(atchley_seq)

    batch = list(zip(input_ids_list, output_labels_list, mask_for_loss_list, atchley_list))

    # 修改：替换 shuffle_and_split_data 为内置 random.shuffle 和手动 split
    random.shuffle(batch)  # shuffle batch list
    val_size = int(len(batch) * 0.1)  # ratio=0.1 for val
    validate_batch = batch[:val_size]
    train_batch = batch[val_size:]

    def unzip_gpt(b):
        if not b: return [], [], [], []
        return zip(*b)

    i_t, o_t, m_t, a_t = unzip_gpt(train_batch)
    i_v, o_v, m_v, a_v = unzip_gpt(validate_batch)

    train_dataset = Dataset_for_beta(i_t, o_t, m_t, a_t)
    if len(i_v) > 0:
        validate_dataset = Dataset_for_beta(i_v, o_v, m_v, a_v)
    else:
        validate_dataset = Dataset_for_beta([], [], [], [])

    return train_dataset, validate_dataset, vocab_size


def make_data_for_seq2seq(data_path, tcr_vocab_path, pep_vocab_path,
                          mhc_allele='HLA-A*02:01', mode='train', max_len=32,
                          e_pmhc_path=None, hla_csv_path=None):
    """
    构建 seq2seq 数据集。

    新增参数
    --------
    e_pmhc_path  : str | None
        e_pmhc.csv 路径（epitope → pmhc pseudo）。
        提供后，每条样本将按抗原肽序列动态匹配对应的 MHC pseudo sequence；
        同一抗原肽若存在多个 MHC，则按每个 MHC 分别生成一条样本（数据扩增）。
    hla_csv_path : str | None
        hla_pseudo_seqs.csv 路径（allele → pseudo-sequence）。
        用于从 mhc_allele 名称解析 pseudo sequence 作为回退值。
    mhc_allele   : str
        当 e_pmhc.csv 中查找不到某抗原肽时，使用此等位基因作为回退 MHC。
    """
    data_df = pd.read_csv(data_path)
    data_df = data_df.dropna(subset=['beta', 'epitope'] if mode == 'train' else ['epitope'])

    tcr_token_df = pd.read_csv(tcr_vocab_path)
    tcr_vocab_freq = {t: f for t, f in zip(tcr_token_df['token'], tcr_token_df['frequency'])}
    tcr_token_list = list(tcr_token_df['token']) + list("RHKDESTNQCUGPAVILMFYW")
    tcr_token2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3, '[UNK]': 4}
    start = len(tcr_token2idx)
    for i, w in enumerate(tcr_token_list):
        if w not in tcr_token2idx: tcr_token2idx[w] = i + start
    tcr_idx2token = {v: k for k, v in tcr_token2idx.items()}
    tcr_vocab_size = len(tcr_token2idx)

    pep_token_df = pd.read_csv(pep_vocab_path)
    pep_vocab_freq = {t: f for t, f in zip(pep_token_df['token'], pep_token_df['frequency'])}
    pep_token_list = list(pep_token_df['token']) + list("RHKDESTNQCUGPAVILMFYW")
    pep_token2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3, '[UNK]': 4}
    start = len(pep_token2idx)
    for i, w in enumerate(pep_token_list):
        if w not in pep_token2idx: pep_token2idx[w] = i + start
    pep_idx2token = {v: k for k, v in pep_token2idx.items()}
    pep_vocab_size = len(pep_token2idx)

    # ---- 动态 MHC 映射（核心新增逻辑）----
    # 若指定路径不存在，自动在 data_path 同目录及上级目录中搜索 e_pmhc.csv
    if e_pmhc_path and not os.path.isfile(e_pmhc_path):
        data_dir = os.path.dirname(os.path.abspath(data_path))
        found = False
        for candidate_dir in [data_dir, os.path.dirname(data_dir)]:
            candidate = os.path.join(candidate_dir, 'e_pmhc.csv')
            if os.path.isfile(candidate):
                print(f"[Data] --e_pmhc_path '{e_pmhc_path}' 不存在，自动使用: {candidate}")
                e_pmhc_path = candidate
                found = True
                break
        if not found:
            print(f"[Warning] e_pmhc_path '{e_pmhc_path}' 不存在，且在 {data_dir} 及其上级目录中未找到 e_pmhc.csv，MHC 动态匹配将被跳过。")

    # 同理，自动搜索 hla_pseudo_seqs.csv
    if hla_csv_path and not os.path.isfile(hla_csv_path):
        data_dir = os.path.dirname(os.path.abspath(data_path))
        for candidate_dir in [data_dir, os.path.dirname(data_dir)]:
            candidate = os.path.join(candidate_dir, 'hla_pseudo_seqs.csv')
            if os.path.isfile(candidate):
                print(f"[Data] --hla_csv_path '{hla_csv_path}' 不存在，自动使用: {candidate}")
                hla_csv_path = candidate
                break

    # epi_pmhc_map: {epitope_str: [pmhc_pseudo_seq, ...]}
    epi_pmhc_map = load_epitope_pmhc_map(e_pmhc_path)

    # 全局回退 pseudo sequence（当 epitope 不在 epi_pmhc_map 中时使用）
    fallback_mhc_pseudo = get_mhc_pseudo(mhc_allele, hla_csv_path)
    if not fallback_mhc_pseudo:
        print(f"[Warning] 回退 MHC allele '{mhc_allele}' 未找到对应 pseudo sequence，编码器前缀将为空。")

    PAD = 0

    def _build_one_sample(tcr_tok_ids, pep_tok_ids, mhc_pseudo):
        """将单条 (TCR, 抗原肽, MHC pseudo) 编码为模型输入格式。"""
        dec_input_ids = [tcr_token2idx['[CLS]']] + tcr_tok_ids
        out_labels    = tcr_tok_ids + [tcr_token2idx['[SEP]']]
        n_pad1 = max_len - len(dec_input_ids)
        n_pad2 = max_len - len(out_labels)
        if n_pad1 > 0: dec_input_ids += [PAD] * n_pad1
        if n_pad2 > 0: out_labels    += [PAD] * n_pad2

        # mask 覆盖真实 token（含 SEP），padding 位为 0
        mask_for_loss = [1] * len(tcr_tok_ids) + [1] + [0] * max(0, n_pad2)

        mhc_ids = [pep_token2idx.get(aa, PAD) for aa in mhc_pseudo]
        enc_input_ids = [pep_token2idx['[CLS]']] + mhc_ids + pep_tok_ids + [pep_token2idx['[SEP]']]
        if len(enc_input_ids) < max_len:
            enc_input_ids += [PAD] * (max_len - len(enc_input_ids))
        else:
            enc_input_ids = enc_input_ids[:max_len]

        enc_atchley = build_atchley_for_ids(enc_input_ids, pep_idx2token, max_len)
        dec_atchley = build_atchley_for_ids(dec_input_ids, tcr_idx2token, max_len)

        # ── 仅抗原肽（epitope-only）的 Atchley 特征 ──────────────────────────
        # enc_input_ids 结构：[CLS] + mhc_ids + pep_tok_ids + [SEP] + [PAD...]
        # 抗原肽 token 起始偏移 = 1 (CLS) + len(mhc_ids)
        epi_start  = 1 + len(mhc_ids)
        epi_end    = epi_start + len(pep_tok_ids)          # 不含 SEP
        # 构造仅含 epitope token 的 ID 序列（其余位置填 PAD=0）
        epi_ids_padded = ([PAD] * epi_start                # CLS + MHC → 置零
                          + enc_input_ids[epi_start:epi_end]  # 仅 epitope
                          + [PAD] * (max_len - epi_end))   # 剩余 → 置零
        epi_ids_padded = epi_ids_padded[:max_len]
        epi_atchley = build_atchley_for_ids(epi_ids_padded, pep_idx2token, max_len)
        # ────────────────────────────────────────────────────────────────────

        return [dec_input_ids, out_labels, mask_for_loss,
                enc_input_ids, enc_atchley, dec_atchley, epi_atchley]

    def process_df_to_dataset(df):
        beta_list    = [str(beta) for beta in df['beta']]
        # 统一归一化：去首尾空白 + 转大写，与 epi_pmhc_map 的键保持一致
        pep_seq_list = [str(pep).strip().upper() for pep in df['epitope']]

        tcr_token_idx_list = [tokenize_with_vocab_optimized(s, tcr_token2idx, tcr_vocab_freq) for s in beta_list]
        pep_token_idx_list = [tokenize_with_vocab_optimized(s, pep_token2idx, pep_vocab_freq) for s in pep_seq_list]

        # 诊断：打印数据集中前3条抗原肽，便于与字典键核对
        sample_epi = list(dict.fromkeys(pep_seq_list))[:3]
        print(f"[Data] 数据集抗原肽示例（前3条，归一化后）: {sample_epi}")

        batch = []
        miss_count  = 0
        miss_sample = []   # 记录未命中的抗原肽（去重，最多展示5条）
        for i, (tcr_tok_ids, pep_tok_ids) in enumerate(zip(tcr_token_idx_list, pep_token_idx_list)):
            epitope_str = pep_seq_list[i]
            # 动态查找该抗原肽对应的所有 MHC pseudo sequence
            pmhc_list = epi_pmhc_map.get(epitope_str)
            if not pmhc_list:
                pmhc_list = [fallback_mhc_pseudo] if fallback_mhc_pseudo else ['']
                miss_count += 1
                if len(miss_sample) < 5 and epitope_str not in miss_sample:
                    miss_sample.append(epitope_str)
            # 每个 MHC 单独生成一条样本（数据扩增）
            for mhc_pseudo in pmhc_list:
                batch.append(_build_one_sample(tcr_tok_ids, pep_tok_ids, mhc_pseudo))

        if miss_count > 0:
            print(f"[Data] {miss_count} 条样本未在 e_pmhc.csv 中找到对应 MHC，已使用回退值 '{mhc_allele}'。")
            print(f"[Data] 未命中抗原肽示例（最多5条）: {miss_sample}")

        if not batch:
            return None
        dec_in, out_lbl, mfl, enc_in, enc_att, dec_att, epi_att = zip(*batch)
        return Dataset_for_seq2seq(
            torch.LongTensor(dec_in), torch.LongTensor(out_lbl), torch.FloatTensor(mfl),
            torch.LongTensor(enc_in), torch.FloatTensor(enc_att), torch.FloatTensor(dec_att),
            torch.FloatTensor(epi_att),
        )

    if mode == 'train':
        # 随机 8:2 拆分（行级别），不再使用 Leave-Group-Out 也不添加负样本
        data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)
        val_size  = int(len(data_df) * 0.2)
        val_df    = data_df.iloc[:val_size].reset_index(drop=True)
        train_df  = data_df.iloc[val_size:].reset_index(drop=True)
        print(f"[Data] 随机拆分完成：训练集 {len(train_df)} 条，验证集 {len(val_df)} 条。")
        train_dataset    = process_df_to_dataset(train_df)
        validate_dataset = process_df_to_dataset(val_df)
        return train_dataset, validate_dataset, tcr_vocab_size, pep_vocab_size, tcr_idx2token, pep_idx2token

    elif mode == 'generate':
        # 每个唯一抗原肽 × 其对应的所有 MHC，分别生成一条推理样本
        # 统一归一化：去首尾空白 + 转大写，与 epi_pmhc_map 的键保持一致
        pep_set = sorted(list(set([str(p).strip().upper() for p in data_df['epitope']])))
        batch   = []
        miss_count  = 0
        miss_sample = []
        print(f"[Data] 推理集抗原肽示例（前3条，归一化后）: {pep_set[:3]}")
        for pep_seq in pep_set:
            pep_tok_ids = tokenize_with_vocab_optimized(pep_seq, pep_token2idx, pep_vocab_freq)
            pmhc_list   = epi_pmhc_map.get(pep_seq)
            if not pmhc_list:
                pmhc_list = [fallback_mhc_pseudo] if fallback_mhc_pseudo else ['']
                miss_count += 1
                if len(miss_sample) < 5 and pep_seq not in miss_sample:
                    miss_sample.append(pep_seq)
            for mhc_pseudo in pmhc_list:
                mhc_ids = [pep_token2idx.get(aa, PAD) for aa in mhc_pseudo]
                enc_input_ids = [pep_token2idx['[CLS]']] + mhc_ids + pep_tok_ids + [pep_token2idx['[SEP]']]
                if len(enc_input_ids) < max_len:
                    enc_input_ids += [PAD] * (max_len - len(enc_input_ids))
                else:
                    enc_input_ids = enc_input_ids[:max_len]
                enc_atchley = build_atchley_for_ids(enc_input_ids, pep_idx2token, max_len)
                batch.append([enc_input_ids, enc_atchley])

        if miss_count > 0:
            print(f"[Data] {miss_count} 种抗原肽未在 e_pmhc.csv 中找到对应 MHC，已使用回退值 '{mhc_allele}'。")
            print(f"[Data] 未命中抗原肽示例（最多5条）: {miss_sample}")

        input_ids, enc_att = zip(*batch)
        dataset = Dataset_for_generate(torch.LongTensor(input_ids), torch.FloatTensor(enc_att))
        return dataset, tcr_idx2token, pep_idx2token, tcr_vocab_size, pep_vocab_size

    else:
        raise ValueError(f"Unknown mode: {mode}")