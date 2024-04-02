# a simple implementation of transformer model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math


def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
    """
    input size of QKV: (batch_size, seq_len, d_*)
    """
    d_k = Q.size(-1)
    K_T = K.transpose(1, 2)
    scores = Q @ K_T / math.sqrt(d_k)
    attn = F.softmax(scores, dim=-1)
    output = attn @ V
    return output


class MultiHeadAttention(nn.Module):
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


    def forward(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        Q_heads = [Q @ self.W_Q[i] for i in range(self.h)]
        K_heads = [K @ self.W_K[i] for i in range(self.h)]
        V_heads = [V @ self.W_V[i] for i in range(self.h)]

        heads = [scaled_dot_product_attention(Q_heads[i], K_heads[i], V_heads[i]) for i in range(self.h)]
        concat_heads = torch.concat(heads, dim=-1)
        output = concat_heads @ self.W_O

        return output

        
class FeedForward(nn.Module):
    def __init__(self, d_ff: int, d_model:int) -> None:
        super().__init__()

        self.ffn_1 =  nn.Linear(in_features=d_model, out_features=d_ff, bias=True)
        self.ffn_2 =  nn.Linear(in_features=d_ff, out_features=d_model, bias=True)
        self.activation = nn.Relu()

    def forwar(self, x: Tensor) -> Tensor:
        x = self.ffn_1(x)
        x = self.activation(x)
        x = self.ffn_2(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.mha = MultiHeadAttention()
        self.ffn = FeedForward()
        self.layer_norm = nn.LayerNorm()

    def forwar(self, x: Tensor) -> Tensor:
        x = self.layer_norm(self.mha(x, x, x) + x)
        x = self.layer_norm(self.ffn(x) + x)
        
        return x


class DecoderBlock(nn.Module):
    def __init__(self) -> None:
        super.__init__()

        self.mha = MultiHeadAttention()
        self.ffn = FeedForward()
        self.layer_norm = nn.LayerNorm()


    def forwar(self, x: Tensor, k: Tensor, v: Tensor) -> Tensor:
        x = self.layer_norm(self.mha(x, x, x) + x)
        x = self.layer_norm(self.mha(x, k, v) + x)
        x = self.layer_norm(self.ffn(x) + x)
        return x

        
class Encoder(nn.Module):
    def __init__(self, n_layer: int) -> None:
        super.__init__()

        self.n_layer = n_layer
        self.layers = nn.ModuleList([EncoderBlock() for _ in range(n_layer)])

    def forward(self, x: Tensor) -> None:
        for layer in self.layers:
            x = layer(x)
        
        return x


class Decoder(nn.Module):
    def __init__(self, n_layer: int) -> None:
        super.__init__()

        self.n_layer = n_layer
        self.layers = nn.ModuleList([DecoderBlock() for _ in range(n_layer)])

    def forward(self, x: Tensor, k: Tensor, v: Tensor) -> None:
        for layer in self.layers:
            x = layer(x, k, v)
        
        return x


class Transformer(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()

        self.h = 8
        self.d_model = 1024
        self.layer = 8
        self.d_k = self.d_v = self.d_model / self.h 

        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x: Tensor) -> None:
        emb = self.encoder(x)
        x = self.decoder(x, emb, emb)

        return x