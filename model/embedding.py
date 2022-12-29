import math
import torch
import torch.nn as nn
from torch.autograd import Variable

from parser import args


class Embeddings(nn.Module):
    """
    两个Embedding层共享相同的权重矩阵。
    使用pre-trained的nn.Embeddings，并且trainable。
    """
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)  # 乘法是扩大数值，减少位置编码影响


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, device=args.device)
        position = torch.arange(0., max_len,  device=args.device).unsqueeze(1)  # （max_len, 1）

        div_term = torch.exp(torch.arange(0., d_model, 2,  device=args.device)
                             *- (math.log(10000.0) / d_model))  # (1. d_model//2)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 扩充维度 [1, max_len, d_model]，一个batch 广播加法

        self.register_buffer('pe', pe)  # 参数保存，但不更新

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)  # pe 第二维度 max_len 大小 进行截取
        return self.dropout(x)
