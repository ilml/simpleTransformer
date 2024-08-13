# simple implementation of attention 

import torch
import math
from torch import nn
from torch.nn import functional as F


class attention(nn.Module):
    def __init__(self, dim, head):
        super().__init__()
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.head = head


    def forward(self, x, mask=None):
        batch_size, seq_len, dim = x.shape
        head_dim = dim // self.head
        xq = self.wq(x).view(batch_size, seq_len, self.head, head_dim).transpose(1, 2)
        xk = self.wk(x).view(batch_size, seq_len, self.head, head_dim).transpose(1, 2)
        xv = self.wv(x).view(batch_size, seq_len, self.head, head_dim).transpose(1, 2)

        score = xq @ xk.transpose(2, 3) / math.sqrt(head_dim)
        if mask is not None:
            score += mask
        attn_weights = F.softmax(score, dim = -1).type_as(xq)
        output = (attn_weights @ xv).transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        output = self.wo(output)

        return output


if __name__ == "__main__":
    batch_size = 4
    seq_len = 4
    dim = 32
    head = 8
    x = torch.rand(batch_size, seq_len, dim)
    mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    attention_layer = attention(dim, head)
    output = attention_layer(x, mask)

