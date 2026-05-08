import torch


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping=False, input_ids=None):
        """
        初始化 BeamHypotheses
        """
        self.num_beams = num_beams
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.input_ids = input_ids
        self.beams = []  # 存储 (score, hypothesis) 的列表
        self.worst_score = 1e9  # 初始化为一个很大的数，随着beam填满会变小

    def __len__(self):
        """
        返回当前 beam 中已完成假设的数量
        """
        return len(self.beams)

    def add(self, hyp, sum_log_probs):
        """
        向 beam 中添加一个新的假设 (hypothesis)
        hyp: 生成的序列 tensor
        sum_log_probs: 对数概率之和
        """
        # 计算经过长度惩罚的分数
        # score = log_prob / (length ** penalty)
        score = sum_log_probs / (hyp.shape[-1] ** self.length_penalty)

        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                # 如果超过了 beam size，排序并移除分数最低的
                # sort by score (item 0)
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                # 删除最低分
                del self.beams[sorted_scores[0][1]]
                # 更新 worst_score 为当前最低分 (即 sorted list 中的第2个，因为第1个刚被删了)
                # 实际上现在的 beams 长度是 num_beams，最差的就是 sorted_scores[1] 对应的那个
                # 重新计算 worst_score
                self.worst_score = sorted([s for s, _ in self.beams])[0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len=None):
        """
        [修复点] 新增 is_done 方法
        判断是否可以提前停止 Beam Search
        """
        # 如果生成的假设数量还不够 beam size，肯定不能停
        if len(self) < self.num_beams:
            return False

        # 如果开启了早停，且凑够了数量，直接停
        if self.early_stopping:
            return True

        # 标准 Beam Search 停止条件：
        # 如果当前最优的“未完成”路径（best_sum_logprobs）
        # 即使除以当前长度（cur_len），得分依然低于
        # 我们已经找到的“已完成”路径中最差的那一条（self.worst_score）
        # 那么就没有必要继续搜了，因为只会越搜越差（长度惩罚通常让长序列分数变低/或者概率乘积变小）

        if cur_len is None:
            cur_len = self.max_length

        # 计算当前搜索前沿的最高潜在分数
        cur_score = best_sum_logprobs / (cur_len ** self.length_penalty)

        # 如果最差的已完成结果 >= 当前最好的潜在结果，说明没戏了，停止
        ret = self.worst_score >= cur_score
        return ret


def expand_inputs(input_ids, input_encodes, beam_size):
    """
    为了 Beam Search，将输入数据复制 beam_size 份
    input_ids: [Batch, Seq] -> [Batch*Beam, Seq]
    input_encodes: [Batch, Seq, Dim] -> [Batch*Beam, Seq, Dim]
    """
    batch_size = input_ids.size(0)

    # Expand input_ids
    # [Batch, 1, Seq] -> [Batch, Beam, Seq] -> [Batch*Beam, Seq]
    expanded_ids = input_ids.unsqueeze(1).expand(batch_size, beam_size, -1).contiguous().view(batch_size * beam_size,
                                                                                              -1)

    # Expand encoder outputs
    seq_len = input_encodes.size(1)
    dim = input_encodes.size(2)
    # [Batch, 1, Seq, Dim] -> [Batch, Beam, Seq, Dim] -> [Batch*Beam, Seq, Dim]
    expanded_encodes = input_encodes.unsqueeze(1).expand(batch_size, beam_size, seq_len, dim).contiguous().view(
        batch_size * beam_size, seq_len, dim)

    return expanded_ids, expanded_encodes