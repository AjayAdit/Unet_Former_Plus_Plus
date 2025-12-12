import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class EarlyContextModule(nn.Module):
    def __init__(
        self,
        in_channels,
        compression_ratio=4,
        num_heads=4,
        window_size=8,
        dropout=0.1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.compressed_channels = in_channels // compression_ratio
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (self.compressed_channels // num_heads) ** -0.5

        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, self.compressed_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.compressed_channels),
            nn.ReLU6(inplace=True)
        )

        self.qkv = nn.Conv2d(self.compressed_channels, self.compressed_channels * 3, 1, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Conv2d(self.compressed_channels, self.compressed_channels, 1, bias=False)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        self._init_relative_position_index()
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.expand = nn.Sequential(
            nn.Conv2d(self.compressed_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        nn.init.constant_(self.gate[3].bias, 1.0)

    def _init_relative_position_index(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        identity = x
        B, C, H, W = x.shape

        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        _, _, Hp, Wp = x.shape

        x_comp = self.compress(x)
        comp_dim = x_comp.shape[1]

        qkv = self.qkv(x_comp)
        q, k, v = rearrange(
            qkv,
            'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d',
            qkv=3, h=self.num_heads, d=comp_dim // self.num_heads,
            ws1=self.window_size, ws2=self.window_size,
            hh=Hp // self.window_size, ww=Wp // self.window_size
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale

        rel_pos_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        rel_pos_bias = rel_pos_bias.view(self.window_size ** 2, self.window_size ** 2, -1)
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()
        attn = attn + rel_pos_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v

        out = rearrange(
            out,
            '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)',
            h=self.num_heads, ws1=self.window_size, ws2=self.window_size,
            hh=Hp // self.window_size, ww=Wp // self.window_size
        )
        out = self.proj(out)

        out = out[:, :, :H, :W]

        out = self.expand(out)
        gate = self.gate(out)

        return identity + gate * out


class ECMLight(nn.Module):
    def __init__(self, in_channels, compression_ratio=8, num_heads=2, window_size=8, dropout=0.1):
        super().__init__()

        self.compressed_channels = in_channels // compression_ratio
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (self.compressed_channels // num_heads) ** -0.5

        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, self.compressed_channels, 1, bias=False),
            nn.BatchNorm2d(self.compressed_channels),
            nn.ReLU6(inplace=True)
        )

        self.qkv = nn.Conv2d(self.compressed_channels, self.compressed_channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(self.compressed_channels, self.compressed_channels, 1, bias=False)

        self.expand = nn.Sequential(
            nn.Conv2d(self.compressed_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        identity = x
        B, C, H, W = x.shape

        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        _, _, Hp, Wp = x_padded.shape

        x_comp = self.compress(x_padded)
        comp_dim = x_comp.shape[1]

        qkv = self.qkv(x_comp)
        q, k, v = rearrange(
            qkv,
            'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d',
            qkv=3, h=self.num_heads, d=comp_dim // self.num_heads,
            ws1=self.window_size, ws2=self.window_size,
            hh=Hp // self.window_size, ww=Wp // self.window_size
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(
            out,
            '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)',
            h=self.num_heads, ws1=self.window_size, ws2=self.window_size,
            hh=Hp // self.window_size, ww=Wp // self.window_size
        )
        out = self.proj(out)
        out = out[:, :, :H, :W]

        out = self.expand(out)
        return identity + torch.sigmoid(self.gate) * out

