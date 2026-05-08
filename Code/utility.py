import numpy as np
import torch

# ==========================================
# 1. Atchley 因子定义
# ==========================================
ATCHLEY_FACTORS = {
    'A': [0.591, -1.302, 0.733, 1.570, -0.146],
    'C': [1.343, 0.465, -0.862, -1.020, -0.255],
    'D': [1.050, 0.302, -3.656, 0.259, -3.242],
    'E': [1.357, -0.105, -1.453, 1.477, 0.113],
    'F': [-1.006, -0.590, 1.891, -0.397, 0.412],
    'G': [0.384, 1.652, 1.330, 1.045, 2.064],
    'H': [0.336, -0.417, -1.673, -1.474, -0.078],
    'I': [-1.239, -0.547, 2.131, 0.393, 0.816],
    'K': [1.831, -0.561, 0.533, -0.277, 1.648],
    'L': [-1.019, -0.987, -1.505, 1.266, -0.912],
    'M': [-0.663, -1.524, 2.219, -1.005, 1.212],
    'N': [0.945, 0.828, 1.299, -0.169, 0.933],
    'P': [0.189, 2.081, -1.628, 0.421, -1.392],
    'Q': [0.931, -0.179, -3.005, -0.503, -1.853],
    'R': [1.538, -0.055, 1.502, 0.440, 2.897],
    'S': [0.228, 1.399, -4.760, 0.670, -2.647],
    'T': [-0.032, 0.326, 2.213, 0.908, 1.313],
    'V': [-1.337, -0.279, -0.544, 1.242, -1.262],
    'W': [-0.595, 0.009, 0.672, -2.128, -0.184],
    'Y': [-0.838, 1.512, 0.068, -0.807, 0.883]
}


def _avg_atchley_for_token(tok: str):
    """
    计算Token的平均Atchley因子
    """
    # 简单的 NaN 检查
    if tok != tok or tok in ('[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]'):
        return [0.0] * 5

    vecs = []
    for ch in tok:
        vecs.append(ATCHLEY_FACTORS.get(ch, [0.0] * 5))

    if not vecs:
        return [0.0] * 5

    return list(np.mean(np.array(vecs, dtype=float), axis=0))


# [新增] 这里补上了 Data_prepare_s.py 需要的函数
def build_atchley_for_ids(ids, idx2token, max_len):
    """
    根据Token ID构建Atchley特征矩阵
    """
    toks = [idx2token.get(i, '[PAD]') for i in ids]
    vecs = [_avg_atchley_for_token(t) for t in toks]
    if len(vecs) < max_len:
        vecs += [[0.0] * 5] * (max_len - len(vecs))
    else:
        vecs = vecs[:max_len]
    return vecs


def get_atchley_factors(seq):
    """旧版接口保留"""
    real_seq = []
    for aa in seq:
        real_seq.append(ATCHLEY_FACTORS.get(aa, [0.0] * 5))
    return np.array(real_seq)


# ==========================================
# 2. 其他计算函数
# ==========================================

def compute_struct_constraint(tcr_seq, epi_seq):
    tcr_vec = _avg_atchley_for_token(tcr_seq)
    epi_vec = _avg_atchley_for_token(epi_seq)
    v1 = torch.tensor(tcr_vec).float()
    v2 = torch.tensor(epi_vec).float()
    if v1.norm() == 0 or v2.norm() == 0: return 0.0
    return torch.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


def compute_novelty(seq, train_seqs_set):
    return 1 if seq not in train_seqs_set else 0


def compute_affinity_score(tcr_seq, epi_seq):
    return compute_struct_constraint(tcr_seq, epi_seq)


def shuffle_and_split_data(batch_data, ratio=0.1):
    import random
    random.shuffle(batch_data)
    split_idx = int(len(batch_data) * ratio)
    validate_data = batch_data[:split_idx]
    train_data = batch_data[split_idx:]
    return train_data, validate_data