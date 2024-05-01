import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.cnn = nn.Conv2d(dim, dim, 5, 1, 2, bias=False, groups=dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x_input = x
        x = self.cnn(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)
        return x + x_input
        #x: b, c, h, d

class DownSample(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, 2, 2, bias=False)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        
        x = x.transpose(1, 3)
        x = self.norm(x)
        x = x.transpose(1, 3)
        x = self.conv(x)

        return x
        
class Model(nn.Module):
    def __init__(self, input_dim, num_layers) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.patch = nn.Conv2d(3, input_dim, 3, 1, bias=False, padding=1)
        self.layers = nn.ModuleList([Block(input_dim) for _ in range(2)]+
                                    [DownSample(input_dim, input_dim*2)]+
                                    [Block(input_dim*2) for _ in range(4)]+
                                    [DownSample(input_dim*2, input_dim*4)]+
                                    [Block(input_dim*4) for _ in range(2)])
        self.norm = nn.LayerNorm(input_dim*4)
        self.head = nn.Linear(input_dim*4, 100, bias=False)

    def forward(self, x):
        x = self.patch(x)

        for layer in self.layers:
            x = layer(x)

        x = x.mean([2, 3])
        x = self.norm(x)
        x = self.head(x)

        return x
        
        