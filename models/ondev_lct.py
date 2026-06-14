import torch
from torch import nn


BN_MOMENTUM = 0.01


class LinearDWS(nn.Module):
    def __init__(self, channels: int, strides: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=strides,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.bn_dw = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn_dw(self.dw(x)))
        x = self.bn_pw(self.pw(x))
        return x


class ResidualLinearBottleneck(nn.Module):
    def __init__(self, channels: int, shrink_ratio: float = 0.5, dws_strides: int = 1) -> None:
        super().__init__()
        inner = max(8, int(channels * shrink_ratio))
        self.shrink = nn.Conv2d(channels, inner, kernel_size=1, bias=False)
        self.bn_shrink = nn.BatchNorm2d(inner, momentum=BN_MOMENTUM)
        self.dws = LinearDWS(inner, strides=dws_strides)
        self.expand = nn.Conv2d(inner, channels, kernel_size=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn_shrink(self.shrink(x)))
        out = self.dws(out)
        out = self.bn_expand(self.expand(out))
        return out + x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.1, proj_drop: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(batch, tokens, channels)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 1.0,
        dropout: float = 0.0,
        attn_drop: float = 0.1,
    ) -> None:
        super().__init__()
        ff = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, attn_drop=attn_drop, proj_drop=dropout)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LCTTokenizer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_conv_layers: int,
        n_residual_blocks: int = 4,
        shrink_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        conv_layers = []
        for i in range(n_conv_layers):
            stride = 2 if i == 0 else 1
            conv_layers.extend(
                [
                    nn.Conv2d(3 if i == 0 else embed_dim, embed_dim, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(embed_dim, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                ]
            )
        self.conv_layers = nn.Sequential(*conv_layers)
        self.bridge = LinearDWS(embed_dim, strides=1)
        self.residuals = nn.Sequential(
            *[
                ResidualLinearBottleneck(embed_dim, shrink_ratio=shrink_ratio, dws_strides=1)
                for _ in range(n_residual_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.bridge(x)
        x = self.residuals(x)
        return x.flatten(2).transpose(1, 2)


class OnDevLCT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        img_size: int = 32,
        in_channels: int = 3,
        embed_dim: int = 128,
        num_heads: int = 4,
        depth: int = 1,
        n_conv_layers: int = 1,
        n_residual_blocks: int = 4,
        mlp_ratio: float = 1.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if img_size <= 0:
            raise ValueError(f"Expected a positive img_size for OnDev-LCT, got {img_size}")
        if in_channels != 3:
            raise ValueError(f"Expected in_channels=3 for OnDev-LCT, got {in_channels}")
        self.tokenizer = LCTTokenizer(
            embed_dim=embed_dim,
            n_conv_layers=n_conv_layers,
            n_residual_blocks=n_residual_blocks,
        )
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_drop=attn_dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attention_pool = nn.Linear(embed_dim, 1)
        self.fc = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        weights = torch.softmax(self.attention_pool(x), dim=1)
        x = torch.matmul(weights.transpose(1, 2), x).squeeze(1)
        return self.fc(x)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def ondev_lct_8_1(num_classes: int, img_size: int = 32, **kwargs) -> OnDevLCT:
    return OnDevLCT(num_classes=num_classes, img_size=img_size, depth=8, n_conv_layers=1, **kwargs)


def ondev_lct_4_1(num_classes: int, img_size: int = 32, **kwargs) -> OnDevLCT:
    return OnDevLCT(num_classes=num_classes, img_size=img_size, depth=4, n_conv_layers=1, **kwargs)