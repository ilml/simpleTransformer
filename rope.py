import torch
import math


def positional_encoding(x):
    # x: [B, S, H], pe: [S, H]
    B, S, H = x.shape
    pe = torch.zeros(S, H)
    i = torch.arange(0, H, 2).float()  #[1, H//2]
    pos = torch.arange(S).float()  #[S, 1]
    div = torch.exp(-math.log(1e4) / H * i)
    angle = torch.outer(pos,div)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    return pe



def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0): 
    theta = torch.exp(math.log(1e4) * torch.arange(0, dim, 2)[:dim//2] / dim)
    m = torch.arange(seq_len)
    freqs = torch.outer(m, theta).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.view(*xq.shape[::-1], -1, 2)
    xk_ = xq.view(*xk.shape[::-1], -1, 2)

    xq_ = xq_.view_as_complex()
    xk_ = xk_.view_as_complex()




if __name__ == "__main__":
    x = torch.ones(4,4,4)
    pe = positional_encoding(x)
    print(pe)
     