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


    


if __name__ == "__main__":
    x = torch.ones(4,4,4)
    pe = positional_encoding(x)
    print(pe)
     