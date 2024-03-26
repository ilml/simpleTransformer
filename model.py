# a simple implementation of transformer model

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    input size of QKV: (batch_size, seq_len, d_*)
    """
    d_k = Q.size(-1)
    K_T = K.transpose(1, 2)
    scores = Q @ K_T / math.sqrt(d_k)
    attn = F.softmax(scores, dim=-1)
    output = attn @ V
    return output


class Transformer(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.h = 8
        self.d_model = 1024
        self.layer = 8
        self.d_k = self.d_v = self.d_model / self.h 
    
    def forward():
        pass


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, h: int, d_model: int, d_k: int, d_v: int) -> None:
        super().__init__()
        
        self.h = h

        self.W_Q = nn.Parameter(torch.empty(h, d_model, d_k))
        self.W_K = nn.Parameter(torch.empty(h, d_model, d_k))
        self.W_V = nn.Parameter(torch.empty(h, d_model, d_v))
        self.W_O = nn.Parameter(torch.empty(h * d_v, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)
        nn.init.xavier_uniform_(self.W_O)


    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        Q_heads = [Q @ self.W_Q[i] for i in range(self.h)]
        K_heads = [K @ self.W_K[i] for i in range(self.h)]
        V_heads = [V @ self.W_V[i] for i in range(self.h)]

        heads = [scaled_dot_product_attention(Q_heads[i], K_heads[i], V_heads[i]) for i in range(self.h)]
        concat_heads = torch.concat(heads, dim=-1)
        output = concat_heads @ self.W_O

        return output

