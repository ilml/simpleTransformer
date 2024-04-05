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


def positional_encoding(seq_len: int, d_model: int) -> Tensor:
    """
    return positional encoding matrix for input data
    """
    pos = torch.arange(seq_len).unsqueeze(1)
    i = torch.arange(0, d_model, 2).unsqueeze(0)
    div = torch.exp(i * (-math.log(1e4)) / d_model)

    pe_even = torch.sin(pos * div)
    pe_odd = torch.cos(pos * div)
    pe = torch.stack([pe_even, pe_odd], dim=-1).flatten(-2)
    
    return pe.unsqueeze(0)



class MultiHeadAttention(nn.Module):
    def __init__(self, h: int, d_model: int, d_k: int, d_v: int) -> None:
        super().__init__()
        
        self.h = h

        # we can use nn.Linear here, but use nn.Paramerter in favor of matmul
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
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.ffn_1(x)
        x = self.activation(x)
        x = self.ffn_2(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, h: int, d_model: int, d_k: int, d_v: int, d_ff: int) -> None:
        super().__init__()

        self.mha = MultiHeadAttention(h=h, d_model=d_model, d_k=d_k, d_v=d_v)
        self.ffn = FeedForward(d_ff=d_ff, d_model=d_model)
        self.ln1 = nn.LayerNorm(d_model)  
        self.ln2 = nn.LayerNorm(d_model)  
        self.dropout = nn.Dropout(0.1) 

    def forward(self, x: Tensor) -> Tensor:
        x = self.ln1(self.dropout(self.mha(x, x, x)) + x)
        x = self.ln2(self.dropout(self.ffn(x)) + x)
        
        return x


class DecoderBlock(nn.Module):
    def __init__(self, h: int, d_model: int, d_k: int, d_v: int, d_ff: int) -> None:
        super().__init__()

        self.m_mha = MultiHeadAttention(h=h, d_model=d_model, d_k=d_k, d_v=d_v)
        self.mha = MultiHeadAttention(h=h, d_model=d_model, d_k=d_k, d_v=d_v)
        self.ffn = FeedForward(d_ff=d_ff, d_model=d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1) 

    def forward(self, x: Tensor, k: Tensor, v: Tensor) -> Tensor:
        x = self.ln1(self.dropout(self.m_mha(x, x, x)) + x)
        x = self.ln2(self.dropout(self.mha(x, k, v)) + x)
        x = self.ln3(self.dropout(self.ffn(x)) + x)
        return x

        
class Encoder(nn.Module):
    def __init__(self, n_layer: int, h: int, d_model: int, d_k: int, d_v: int, d_ff: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList([EncoderBlock(h=h, d_model=d_model, d_k=d_k, d_v=d_v, d_ff=d_ff) for _ in range(n_layer)])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        
        return x


class Decoder(nn.Module):
    def __init__(self, n_layer: int, h: int, d_model: int, d_k: int, d_v: int, d_ff: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(h=h, d_model=d_model, d_k=d_k, d_v=d_v, d_ff=d_ff) for _ in range(n_layer)])

    def forward(self, x: Tensor, k: Tensor, v: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, k, v)
        
        return x


class Transformer(nn.Module):
    def __init__(self, n_layer: int = 6, h: int = 8, d_model: int = 512, d_ff: int = 2048, vocab_size: int = 37000) -> None:
        super().__init__()

        d_k = d_v = d_model // h 
        self.d_model = d_model

        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.output_linear = nn.Linear(d_model, vocab_size)
        self.output_linear.weight = self.emb.weight
        self.dropout = nn.Dropout(0.1) 

        self.encoder = Encoder(n_layer=n_layer, h=h, d_model=d_model, d_k=d_k, d_v=d_v, d_ff=d_ff)
        self.decoder = Decoder(n_layer=n_layer, h=h, d_model=d_model, d_k=d_k, d_v=d_v, d_ff=d_ff)
    
    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        src_emb = self.dropout(self.emb(src) * math.sqrt(self.d_model) + positional_encoding(src.shape[1], self.d_model))
        tgt_emb = self.dropout(self.emb(tgt) * math.sqrt(self.d_model) + positional_encoding(tgt.shape[1], self.d_model))

        src_emb = self.encoder(src_emb)
        output = self.decoder(tgt_emb, src_emb, src_emb)
        logits = self.output_linear(output)
        
        return F.log_softmax(logits, dim=-1)
        
if __name__ == "__main__":
    model = Transformer()
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")