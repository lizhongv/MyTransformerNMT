import torch
import math, copy
import torch.nn.functional as F
import torch.nn as nn

from utils import clones


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'."
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 保证落入激活函数正常区

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # 0对应处转换负无穷，经exp后为0 
    p_attn = F.softmax(scores, dim=-1)  # 按最后一维度，activation function, mapped to 0~1
    
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn  # value加权和，注意力矩阵 


class MultiHeadedAttention(nn.Module):
    "Take in model size and number of heads."
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  # 保证可以整除
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # Wq, Wk, Wv 和 Wz
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)  # 多头
        nbatches = query.size(0)

        query, key, value = \
            [ l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)  # concat
        return self.linears[-1](x)
