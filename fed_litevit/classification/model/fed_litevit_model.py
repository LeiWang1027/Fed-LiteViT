import torch
import torch.nn as nn
from timm.models.vision_transformer import trunc_normal_
from timm.layers import SqueezeExcite, DropPath
from fed_litevit.classification.model.swiftformer import SwiftFormerEncoder, ConvEncoder, SwiftFormerEncoder_nlr
from fed_litevit.classification.model.SHI.SRResNet_class import MFEblock, oneConv, ASPPConv
from fed_litevit.classification.model.old_fed_litevit import CascadedGroupAttention as CGA
from fed_litevit.classification.model.dream_code_v2.EMAttention import EMA
from fed_litevit.classification.model.dream_code_v2.MLLAttention import LinearAttention, MLLABlock

def stem(in_chs, out_chs):
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_chs // 2),
        nn.ReLU(),
        nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_chs),
        nn.ReLU(),
    )

class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)
        self.dilation = dilation
    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps)**0.5
        m = nn.Conv2d(w.size(1) * self.c.groups, w.size(0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class BN_Linear(nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', nn.BatchNorm1d(a))
        self.add_module('l', nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            nn.init.constant_(self.l.bias, 0)
    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim, input_resolution):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0, resolution=input_resolution)
        self.act = nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim, resolution=input_resolution)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0, resolution=input_resolution // 2)
    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x

class Residual(nn.Module):
    def __init__(self, m, drop=0.2):
        super().__init__()
        self.m = m
        self.drop = drop
    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1, device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

class FFN(nn.Module):
    def __init__(self, ed, h, resolution):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h, resolution=resolution)
        self.act = nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0, resolution=resolution)
    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x

class LightASPPConv(nn.Module):
    def __init__(self, in_c, out_c, rate):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=rate, dilation=rate, groups=in_c),
            nn.Conv2d(in_c, out_c, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)

class CascadedGroupAttention(nn.Module):
    def __init__(self, dim, key_dim, num_heads=8, attn_ratio=4, resolution=14, kernels=[5, 5, 5, 5]):
        super().__init__()
        self.num_heads = num_heads
        self.d = int(attn_ratio * key_dim)
        self.sw_dim = dim // num_heads
        self.mfeb = LinAtt(self.sw_dim, attn_ratio, resolution, num_heads)
        self.proj = nn.Sequential(nn.ReLU(), Conv2d_BN(self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

    def forward(self, x):
        feats_in = x.chunk(self.num_heads, dim=1)
        feats_out = []
        feat = feats_in[0]
        for i in range(self.num_heads):
            if i > 0:
                feat = feat + feats_in[i]
            feat = self.mfeb(feat)
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, 1))
        return x


class ParallelGroupAttention(nn.Module):
    """Parallel ablation of CascadedGroupAttention with shared LinAtt."""

    def __init__(self, dim, key_dim, num_heads=8, attn_ratio=4, resolution=14, kernels=[5, 5, 5, 5]):
        super().__init__()
        self.num_heads = num_heads
        self.d = int(attn_ratio * key_dim)
        self.sw_dim = dim // num_heads
        self.mfeb = LinAtt(self.sw_dim, attn_ratio, resolution, num_heads)
        self.proj = nn.Sequential(nn.ReLU(), Conv2d_BN(self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

    def forward(self, x):
        feats_in = x.chunk(self.num_heads, dim=1)
        feats_out = []
        for i in range(self.num_heads):
            feat = self.mfeb(feats_in[i])
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, 1))
        return x

class LinAtt(nn.Module):
    def __init__(self, ed, ar=3, resolution=14, num_heads=8):
        super().__init__()
        in_resolution = (resolution, resolution)
        self.sw_attn = LinearAttention(ed, in_resolution, num_heads, True)
    def forward(self, x):
        y = self.sw_attn(x)
        B, L, C = y.shape
        H = W = int(L**0.5)
        y = y.reshape(B, C, H, W)
        return y

class MFEblock_Globa(nn.Module):
    def __init__(self, ed, ar=3, resolution=14, num_heads=8):
        super().__init__()
        out_channels = ed
        atrous_rates = [1,3,6]
        rate3 = atrous_rates[2]
        self.layer1 = nn.Sequential(
            nn.Conv2d(ed, ed, 3, padding=1, groups=ed),
            nn.Conv2d(ed, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.layer4 = LightASPPConv(ed, out_channels, rate3)
        self.project = nn.Sequential(
            nn.Conv2d(ed, ed//2, 1),
            nn.BatchNorm2d(ed//2),
            nn.ReLU(),
            nn.Conv2d(ed//2, ed, 1)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=2)
        self.softmax_1 = nn.Sigmoid()
        self.SE_shared = oneConv(ed,ed,1,0,1)
    def forward(self, x):
        y0 = self.layer1(x)
        y3 = self.layer4(y0+x)
        y0_weight = self.SE_shared(self.gap(y0))
        y3_weight = self.SE_shared(self.gap(y3))
        weight = torch.cat([y0_weight,y3_weight],2)
        weight = self.softmax(self.softmax_1(weight))
        y0_weight = torch.unsqueeze(weight[:,:,0],2)
        y3_weight = torch.unsqueeze(weight[:,:,1],2)
        x_att = y0_weight*y0+y3_weight*y3
        return self.project(x_att+x)

class LocalWindowAttention(nn.Module):
    def __init__(self, dim, key_dim, num_heads=8, attn_ratio=4, resolution=14, window_resolution=7, kernels=[5, 5, 5, 5], use_parallel_attention=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        self.window_resolution = window_resolution
        window_resolution = min(window_resolution, resolution)
        attention_class = ParallelGroupAttention if use_parallel_attention else CascadedGroupAttention
        self.attn = attention_class(dim, key_dim, num_heads, attn_ratio=attn_ratio, resolution=window_resolution, kernels=kernels)
    def forward(self, x):
        H = W = self.resolution
        B, C, H_, W_ = x.shape
        assert H == H_ and W == W_
        if H <= self.window_resolution and W <= self.window_resolution:
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 1)
            pad_b = (self.window_resolution - H % self.window_resolution) % self.window_resolution
            pad_r = (self.window_resolution - W % self.window_resolution) % self.window_resolution
            if pad_b > 0 or pad_r > 0:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))
            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_resolution
            nW = pW // self.window_resolution
            x = x.view(B, nH, self.window_resolution, nW, self.window_resolution, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_resolution, self.window_resolution, C).permute(0, 3, 1, 2)
            x = self.attn(x)
            x = x.permute(0, 2, 3, 1).view(B, nH, nW, self.window_resolution, self.window_resolution, C).transpose(2, 3).reshape(B, pH, pW, C)
            if pad_b > 0 or pad_r > 0:
                x = x[:, :H, :W].contiguous()
            x = x.permute(0, 3, 1, 2)
        return x

class GlobalAttention(nn.Module):
    def __init__(self, dim, key_dim, num_heads=8, attn_ratio=4, resolution=14, window_resolution=7, kernels=[5, 5, 5, 5], em_dim=None, use_parallel_attention=False, use_mfe=True):
        super().__init__()
        self.use_mfe = use_mfe
        if self.use_mfe:
            self.mfeb_globa = MFEblock_Globa(dim, attn_ratio, resolution, num_heads)
        window_resolution = min(window_resolution, resolution)
        attention_class = ParallelGroupAttention if use_parallel_attention else CascadedGroupAttention
        self.attn = attention_class(dim, key_dim, num_heads, attn_ratio=attn_ratio, resolution=window_resolution, kernels=kernels)
    def forward(self, x):
        if self.use_mfe:
            x = self.mfeb_globa(x)
        x = self.attn(x)
        return x

class FedLiteViTBlock(nn.Module):
    def __init__(self, type, ed, kd, nh=8, ar=4, resolution=14, window_resolution=7, kernels=[5, 5, 5, 5], use_global_attention=False, use_parallel_attention=False, use_global_mfe=True, branch_mode='both', micla_right_groups=8):
        super().__init__()
        if branch_mode not in ['both', 'left_only', 'right_only', 'none']:
            raise ValueError(f"Unsupported branch_mode: {branch_mode}")

        self.branch_mode = branch_mode
        self._save_features = False
        self._cached_left = None
        self._cached_right = None
        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn0 = Residual(FFN(ed, int(ed * 2), resolution))
        if self.branch_mode not in ['left_only', 'none']:
            self.right_path = nn.Sequential(
                nn.Conv2d(ed, ed, kernel_size=1, groups=micla_right_groups),
                nn.SiLU()
            )
        if self.branch_mode not in ['right_only', 'none']:
            self.left_path = nn.Sequential(
                nn.Conv2d(ed, ed, kernel_size=1),
                nn.Conv2d(ed, ed, kernel_size=3, padding=1, groups=ed),
                nn.Conv2d(ed, ed, kernel_size=1),
                nn.SiLU()
            )
        if type == 's' and self.branch_mode != 'none':
            if use_global_attention:
                self.mixer = Residual(GlobalAttention(ed, kd, nh, attn_ratio=ar, resolution=resolution, window_resolution=window_resolution, kernels=kernels, use_parallel_attention=use_parallel_attention, use_mfe=use_global_mfe))
            else:
                self.mixer = Residual(LocalWindowAttention(ed, kd, nh, attn_ratio=ar, resolution=resolution, window_resolution=window_resolution, kernels=kernels, use_parallel_attention=use_parallel_attention))
        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn1 = Residual(FFN(ed, int(ed * 2), resolution))

    def enable_feature_cache(self):
        self._save_features = True

    def disable_feature_cache(self):
        self._save_features = False
        self._cached_left = None
        self._cached_right = None

    def get_cached_features(self):
        return self._cached_left, self._cached_right

    def forward(self, x):
        x = self.dw0(x)
        x = self.ffn0(x)
        right_path = self.right_path(x) if self.branch_mode not in ['left_only', 'none'] else None
        left_path = self.left_path(x) if self.branch_mode not in ['right_only', 'none'] else None
        if left_path is not None:
            left_path = self.mixer(left_path)
        if self._save_features:
            self._cached_left = left_path.detach().cpu() if left_path is not None else None
            self._cached_right = right_path.detach().cpu() if right_path is not None else None
        if self.branch_mode == 'left_only':
              merged = nn.SiLU()(left_path)
        elif self.branch_mode == 'right_only':
              merged = nn.SiLU()(right_path)
        elif self.branch_mode == 'none':
            merged = None
        else:
            merged = nn.SiLU()(right_path * left_path)
        if merged is not None:
            x = x + merged
        x = self.dw1(x)
        x = self.ffn1(x)
        return x

class FedLiteViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 stages=['s', 's', 's'], embed_dim=[64, 128, 192], key_dim=[16, 16, 16],
                 depth=[1, 2, 3], num_heads=[4, 4, 4], window_size=[7, 7, 7], kernels=[5, 5, 5, 5],
                 down_ops=[['subsample', 2],  ['subsample', 2], ['']], distillation=False, use_global_attention=False, use_parallel_attention=False, use_global_mfe=True, branch_mode='both', micla_right_groups=8):
        super().__init__()
        resolution = img_size
        self.patch_embed = nn.Sequential(
            Conv2d_BN(in_chans, embed_dim[0] // 8, 3, 2, 1, resolution=resolution), nn.GELU(),
            Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1, resolution=resolution // 2), nn.GELU(),
            Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1, resolution=resolution // 4), nn.GELU(),
            Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1, resolution=resolution // 8)
        )
        resolution = img_size // patch_size
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        self.blocks = [[], [], [], []]
        for i, (stg, ed, kd, dpth, nh, ar, wd, do) in enumerate(zip(stages, embed_dim, key_dim, depth, num_heads, attn_ratio, window_size, down_ops)):
            for d in range(dpth):
                block = FedLiteViTBlock(stg, ed, kd, nh, ar, resolution, wd, kernels, use_global_attention=(i % 2 == 1), use_parallel_attention=use_parallel_attention, use_global_mfe=use_global_mfe, branch_mode=branch_mode, micla_right_groups=micla_right_groups)
                self.blocks[i].append(block)
            if do[0] == 'subsample':
                resolution_ = (resolution - 1) // do[1] + 1
                self.blocks[i+1].append(nn.Sequential(
                    Residual(Conv2d_BN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i], resolution=resolution)),
                    Residual(FFN(embed_dim[i], int(embed_dim[i] * 2), resolution))
                ))
                self.blocks[i+1].append(PatchMerging(*embed_dim[i:i + 2], resolution))
                resolution = resolution_
                self.blocks[i+1].append(nn.Sequential(
                    Residual(Conv2d_BN(embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1, groups=embed_dim[i + 1], resolution=resolution)),
                    Residual(FFN(embed_dim[i + 1], int(embed_dim[i + 1] * 2), resolution))
                ))
        self.blocks1 = nn.Sequential(*self.blocks[0])
        self.blocks2 = nn.Sequential(*self.blocks[1])
        self.blocks3 = nn.Sequential(*self.blocks[2])
        self.blocks4 = nn.Sequential(*self.blocks[3])
        self.head = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.distillation = distillation
        if distillation:
            self.head_dist = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
            return x
        return self.head(x)
