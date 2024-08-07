import torch
import math


def positional_encoding(x):
    # x: [B, S, H], pe: [S, H]
    B, S, H = x.shape
    pe = torch.zeros(S, H)
    i = torch.arange(0, H, 2).float()  #[1, H//2]
    pos = torch.arange(S).unsqueeze(1).float()  #[S, 1]
    div = torch.exp(-math.log(1e4) / H * i)
    pe[:, 0::2] = torch.sin(pos*div)
    pe[:, 1::2] = torch.cos(pos*div)
    return pe



if __name__ == "__main__":
    x = torch.ones(4,4,4)
    pe = positional_encoding(x)
     