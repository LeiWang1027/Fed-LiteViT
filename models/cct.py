import torch
from torch import nn


class CCTTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        conv_channels=(64, 128),
        kernel_size: int = 3,
        pool_kernel_size: int = 3,
        pool_stride: int = 2,
        pool_padding: int = 1,
    ) -> None:
        super().__init__()
        layers = []
        current_in = in_channels
        for current_out in conv_channels:
            layers.extend(
                [
                    nn.Conv2d(current_in, current_out, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding),
                ]
            )
            current_in = current_out
        self.net = nn.Sequential(*layers)
        self.embedding_dim = current_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x.flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch, tokens, channels)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CCT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        img_size: int = 32,
        in_channels: int = 3,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 1.0,
        dropout: float = 0.0,
        n_conv_layers: int = 2,
    ) -> None:
        super().__init__()
        if img_size <= 0:
            raise ValueError(f"Expected a positive img_size for CCT, got {img_size}")
        if in_channels != 3:
            raise ValueError(f"Expected in_channels=3 for CCT, got {in_channels}")
        if n_conv_layers != 2:
            raise ValueError(f"CCT-4/2 expects n_conv_layers=2, got {n_conv_layers}")
        self.tokenizer = CCTTokenizer(in_channels=in_channels, conv_channels=(64, embed_dim))
        with torch.no_grad():
            seq_len = self.tokenizer(torch.zeros(1, in_channels, img_size, img_size)).shape[1]
        self.positional_embedding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerEncoder(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.attention_pool = nn.Linear(embed_dim, 1)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x)
        x = self.dropout(x + self.positional_embedding)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        pool_weights = torch.softmax(self.attention_pool(x), dim=1)
        x = torch.matmul(pool_weights.transpose(1, 2), x).squeeze(1)
        return self.head(x)


def cct_4_2(num_classes: int, img_size: int = 32, **kwargs) -> CCT:
    return CCT(num_classes=num_classes, img_size=img_size, depth=4, n_conv_layers=2, **kwargs)