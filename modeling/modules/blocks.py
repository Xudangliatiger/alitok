import torch
import torch.nn as nn
from collections import OrderedDict
import einops
from einops.layers.torch import Rearrange


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm
        ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def attention(
            self,
            x: torch.Tensor
    ):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(
            self,
            x: torch.Tensor,
    ):
        attn_output = self.attention(x=self.ln_1(x))
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class UViTBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    