import copy
import torch.nn as nn
import numpy as np
import torch


def clones(module, N):
    """克隆后各模型参数不共享""""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 处理测试数据
def seq_padding(X, padding=0):
    "以一个batch中最长seq的长度进行padding, 返回 ndarray 形式。"
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')  # 右上角全1(不含主对角线), 左下角全0
    return torch.from_numpy(subsequent_mask) == 0  # 0处全True，1处全False
