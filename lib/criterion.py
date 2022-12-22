import torch
import torch.nn as nn
from torch.autograd import Variable


class LabelSmoothing(nn.Module):
    """
    LabelSmoothing 防止过于相信label，添加一个均分分布的噪声。
    """
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing  # 平滑系数e
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
       
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))  # 填充 e*u(k) 其中 u(k)=1/(size-2) 
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)  # ground truth 分布 q(k), q'(k) = (1-e)*q(k) + e*u(k)
        true_dist[:, self.padding_idx] = 0  
      
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
