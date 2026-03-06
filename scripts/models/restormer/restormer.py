# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from abc import abstractmethod
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from models.varnet import CoilSensitivityModel_DCAE
from monai.apps.reconstruction.networks.nets.utils import divisible_pad_t, inverse_divisible_pad_t
from monai.data.fft_utils import fftn_centered, ifftn_centered
from monai.networks.utils import copy_model_state
from timm.layers import DropPath
from torch import Tensor, einsum
from torch.utils.checkpoint import checkpoint
from utils import *


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=timesteps.device
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob, labels_embed_magnitude_scale=1.0):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.labels_embed_magnitude_scale = labels_embed_magnitude_scale

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings * self.labels_embed_magnitude_scale


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


def to(x):
    return {"device": x.device, "dtype": x.dtype}


def pair(x):
    return (x, x) if not isinstance(x, tuple) else x


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim=2)
    flat_x = rearrange(x, "b l c -> b (l c)")
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x


def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum("b x y d, r d -> b x y r", q, rel_k)
    logits = rearrange(logits, "b x y r -> (b x) y r")
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim=2, k=r)
    return logits


class RelPosEmb(nn.Module):
    def __init__(self, block_size, rel_size, dim_head):
        super().__init__()
        height = width = rel_size
        scale = dim_head**-0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, "b (x y) c -> b x y c", x=block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, "b x i y j-> b (x y) (i j)")

        q = rearrange(q, "b x y d -> b y x d")
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, "b x i y j -> b (y x) (j i)")
        return rel_logits_w + rel_logits_h


class OCAB(nn.Module):
    def __init__(
        self,
        dim,
        window_size=8,
        overlap_ratio=0.5,
        num_heads=2,
        dim_head=16,
        bias=False,
    ):
        super(OCAB, self).__init__()
        self.num_spatial_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.dim_head = dim_head
        self.inner_dim = self.dim_head * self.num_spatial_heads
        self.scale = self.dim_head**-0.5

        self.unfold = nn.Unfold(
            kernel_size=(self.overlap_win_size, self.overlap_win_size),
            stride=window_size,
            padding=(self.overlap_win_size - window_size) // 2,
        )
        self.qkv = nn.Conv2d(self.dim, self.inner_dim * 3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=bias)
        self.rel_pos_emb = RelPosEmb(
            block_size=window_size,
            rel_size=window_size + (self.overlap_win_size - window_size),
            dim_head=self.dim_head,
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        qs, ks, vs = qkv.chunk(3, dim=1)

        # spatial attention
        qs = rearrange(
            qs,
            "b c (h p1) (w p2) -> (b h w) (p1 p2) c",
            p1=self.window_size,
            p2=self.window_size,
        )
        ks, vs = map(lambda t: self.unfold(t), (ks, vs))
        ks, vs = map(lambda t: rearrange(t, "b (c j) i -> (b i) j c", c=self.inner_dim), (ks, vs))

        # print(f'qs.shape:{qs.shape}, ks.shape:{ks.shape}, vs.shape:{vs.shape}')
        # split heads
        qs, ks, vs = map(
            lambda t: rearrange(t, "b n (head c) -> (b head) n c", head=self.num_spatial_heads),
            (qs, ks, vs),
        )

        # attention
        qs = qs * self.scale
        spatial_attn = qs @ ks.transpose(-2, -1)
        spatial_attn += self.rel_pos_emb(qs)
        spatial_attn = spatial_attn.softmax(dim=-1)

        out = spatial_attn @ vs

        out = rearrange(
            out,
            "(b h w head) (p1 p2) c -> b (head c) (h p1) (w p2)",
            head=self.num_spatial_heads,
            h=h // self.window_size,
            w=w // self.window_size,
            p1=self.window_size,
            p2=self.window_size,
        )

        # merge spatial and channel
        out = self.project_out(out)

        return out


class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            padding=1,
            groups=channels * 3,
            bias=False,
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor, emb_channels=None):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_channels * 2,
            bias=False,
        )
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

        if emb_channels is not None:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    emb_channels,
                    hidden_channels,
                ),
            )
        else:
            self.emb_layers = None

    def forward(self, x, emb=None):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        if emb is not None and self.emb_layers is not None:
            emb_out = self.emb_layers(emb)
            while len(emb_out.shape) < len(x.shape):
                emb_out = emb_out[..., None]
            x = x + emb_out

        x = self.project_out(x)
        return x


class MoE(nn.Module):
    """
    A sparse Mixture of Experts (MoE) layer with top-k gating.

    This layer routes each input to a specific number of experts (`top_k`)
    and computes a weighted sum of their outputs.
    """

    def __init__(self, channels, expansion_factor, num_experts=4, top_k=1):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Expert networks
        self.experts = nn.ModuleList([GDFN(channels, expansion_factor) for _ in range(self.num_experts)])

        # Gating network
        self.gate = nn.Linear(channels, self.num_experts)

    def forward(self, x):
        b, c, h, w = x.shape

        # 1. Use global average pooling to get a feature vector for routing
        gating_input = F.adaptive_avg_pool2d(x, (1, 1)).view(b, -1)

        # 2. Get router logits from the gate
        router_logits = self.gate(gating_input)  # Shape: (b, num_experts)

        # 3. Select top-k experts
        # Find the top_k scores and their indices
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)  # Shape: (b, top_k)

        # 4. Re-normalize the weights of the top-k experts
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # 5. Create a sparse tensor of the final weights
        # Initialize a tensor of zeros with the same shape as the router logits
        sparse_weights = torch.zeros_like(router_logits)
        # Place the calculated top-k weights at the correct indices
        sparse_weights.scatter_(-1, top_k_indices, top_k_weights)  # In-place scatter

        # 6. Compute and combine expert outputs
        # Note: This implementation computes all experts but only uses the top-k ones.
        # For true computational savings, a more complex dispatching mechanism is needed.
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        # Reshape weights for broadcasting: (b, num_experts) -> (b, num_experts, 1, 1, 1)
        routing_weights = sparse_weights.view(b, self.num_experts, 1, 1, 1)

        # Compute the weighted sum of expert outputs (only top-k will have non-zero weights)
        weighted_output = torch.sum(routing_weights * expert_outputs, dim=1)

        return weighted_output


class TransformerBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        num_heads,
        expansion_factor,
        num_experts=None,
        top_k=1,
        drop_path=0.0,
        norm_type=None,
        post_norm=False,
        emb_channels=None,
    ):
        super(TransformerBlock, self).__init__()

        self.norm_type = norm_type
        if self.norm_type == "ln":
            self.norm1 = nn.LayerNorm(channels)
            self.norm2 = nn.LayerNorm(channels)
        elif self.norm_type == "bn":
            self.norm1 = nn.BatchNorm2d(channels)
            self.norm2 = nn.BatchNorm2d(channels)
        elif self.norm_type is None:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        else:
            raise NotImplementedError("Norm {} is not implemented. (bn or ln or null)".format(norm_type))
        self.attn = MDTA(channels, num_heads)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Conditionally use MoE or a single GDFN
        if num_experts and num_experts > 0:
            self.ffn = MoE(channels, expansion_factor, num_experts, top_k)
        else:
            self.ffn = GDFN(channels, expansion_factor, emb_channels=emb_channels)

        self.post_norm = post_norm

    def forward(self, x, emb=None):
        b, c, h, w = x.shape
        if self.norm_type == "ln":
            if self.post_norm:
                x = x + self.norm1(
                    self.drop_path(self.attn(x)).reshape(b, c, -1).transpose(-2, -1).contiguous()
                ).transpose(-2, -1).contiguous().reshape(b, c, h, w)
                x = x + self.norm2(
                    self.drop_path(self.ffn(x, emb)).reshape(b, c, -1).transpose(-2, -1).contiguous()
                ).transpose(-2, -1).contiguous().reshape(b, c, h, w)
            else:
                x = x + self.drop_path(
                    self.attn(
                        self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
                        .transpose(-2, -1)
                        .contiguous()
                        .reshape(b, c, h, w)
                    )
                )
                x = x + self.drop_path(
                    self.ffn(
                        self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
                        .transpose(-2, -1)
                        .contiguous()
                        .reshape(b, c, h, w),
                        emb,
                    )
                )
        else:
            if self.post_norm:
                x = x + self.norm1(self.drop_path(self.attn(x)))
                x = x + self.norm2(self.drop_path(self.ffn(x, emb)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x)))
                x = x + self.drop_path(self.ffn(self.norm2(x), emb))
        return x


class HybridTransformerBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_channel_heads,
        num_spatial_heads,
        expansion_factor,
        num_experts=None,
        top_k=1,
        drop_path=0.0,
        norm_type=None,
        post_norm=False,
    ):
        super(HybridTransformerBlock, self).__init__()

        self.norm_type = norm_type
        if self.norm_type == "ln":
            self.norm1 = nn.LayerNorm(channels)
            self.norm2 = nn.LayerNorm(channels)
            self.norm3 = nn.LayerNorm(channels)
            self.norm4 = nn.LayerNorm(channels)
        elif self.norm_type == "bn":
            self.norm1 = nn.BatchNorm2d(channels)
            self.norm2 = nn.BatchNorm2d(channels)
            self.norm3 = nn.BatchNorm2d(channels)
            self.norm4 = nn.BatchNorm2d(channels)
        elif self.norm_type is None:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            self.norm3 = nn.Identity()
            self.norm4 = nn.Identity()
        else:
            raise NotImplementedError("Norm {} is not implemented. (bn or ln or null)".format(norm_type))
        self.channel_attn = MDTA(channels, num_channel_heads)
        self.spatial_attn = OCAB(channels, num_heads=num_spatial_heads)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Conditionally use MoE or a single GDFN
        if num_experts and num_experts > 0:
            self.channel_ffn = MoE(channels, expansion_factor, num_experts, top_k)
            self.spatial_ffn = MoE(channels, expansion_factor, num_experts, top_k)
        else:
            self.channel_ffn = GDFN(channels, expansion_factor)
            self.spatial_ffn = GDFN(channels, expansion_factor)

        self.post_norm = post_norm

    def forward(self, x):
        b, c, h, w = x.shape
        if self.norm_type == "ln":
            if self.post_norm:
                x = x + self.norm1(
                    self.drop_path(self.channel_attn(x)).reshape(b, c, -1).transpose(-2, -1).contiguous()
                ).transpose(-2, -1).contiguous().reshape(b, c, h, w)
                x = x + self.norm2(
                    self.drop_path(self.channel_ffn(x)).reshape(b, c, -1).transpose(-2, -1).contiguous()
                ).transpose(-2, -1).contiguous().reshape(b, c, h, w)
                x = x + self.norm3(
                    self.drop_path(self.spatial_attn(x)).reshape(b, c, -1).transpose(-2, -1).contiguous()
                ).transpose(-2, -1).contiguous().reshape(b, c, h, w)
                x = x + self.norm4(
                    self.drop_path(self.spatial_ffn(x)).reshape(b, c, -1).transpose(-2, -1).contiguous()
                ).transpose(-2, -1).contiguous().reshape(b, c, h, w)
            else:
                x = x + self.drop_path(
                    self.channel_attn(
                        self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
                        .transpose(-2, -1)
                        .contiguous()
                        .reshape(b, c, h, w)
                    )
                )
                x = x + self.drop_path(
                    self.channel_ffn(
                        self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
                        .transpose(-2, -1)
                        .contiguous()
                        .reshape(b, c, h, w)
                    )
                )
                x = x + self.drop_path(
                    self.spatial_attn(
                        self.norm3(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
                        .transpose(-2, -1)
                        .contiguous()
                        .reshape(b, c, h, w)
                    )
                )
                x = x + self.drop_path(
                    self.spatial_ffn(
                        self.norm4(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
                        .transpose(-2, -1)
                        .contiguous()
                        .reshape(b, c, h, w)
                    )
                )
        else:
            if self.post_norm:
                x = x + self.norm1(self.drop_path(self.channel_attn(x)))
                x = x + self.norm2(self.drop_path(self.channel_ffn(x)))
                x = x + self.norm3(self.drop_path(self.spatial_attn(x)))
                x = x + self.norm4(self.drop_path(self.spatial_ffn(x)))
            else:
                x = x + self.drop_path(self.channel_attn(self.norm1(x)))
                x = x + self.drop_path(self.channel_ffn(self.norm2(x)))
                x = x + self.drop_path(self.spatial_attn(self.norm3(x)))
                x = x + self.drop_path(self.spatial_ffn(self.norm4(x)))
        return x


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Restormer(nn.Module):
    """
    A Restormer-style U-Net backbone that supports an arbitrary number of levels.

    Args:
        in_channel (int): input channels.
        out_channel (int): output channels.
        num_blocks (List[int]): number of TransformerBlocks per level (length L).
        num_heads  (List[int]): attention heads per level (length L).
        channels   (List[int]): channel width per level from top to bottom (length L, L >= 1).
        num_refinement (int): number of TransformerBlocks in final refinement at top level.
        expansion_factor, num_experts, top_k, drop_path, norm_type, post_norm: passed to TransformerBlock.

    cas_skips contract (same as original):
        - In forward(cas_skips=...), cas_skips is a list/tuple of length L-1.
        - cas_skips[i-1] is added to the encoder *input* at level i (i = 1..L-1).

    Returns:
        out: final image-like tensor.
        cas_outs (tuple of length L-1):
            [dec_out(level=1), ..., dec_out(level=L-2), enc_out(level=L-1)]
        For L=4, this matches original behavior: (out_dec2, out_dec3, out_enc4).
        For L=1, this is (out_dec1).
    """

    def __init__(
        self,
        in_channel=2,
        out_channel=2,
        num_blocks=[4, 6, 6, 8],
        num_heads=[1, 2, 4, 8],
        channels=[48, 96, 192, 384],
        num_refinement=4,
        expansion_factor=2.66,
        num_experts=None,
        top_k=1,
        drop_path=0.0,
        norm_type=None,
        post_norm=False,
        ms_refinement=False,
        time_cond=False,
        label_cond=False,
        num_classes=9,
        timestep_scale=50.0,
        label_class_scale=10.0,
        time_embed_scale=1,
        class_dropout_prob=0.1,
        labels_embed_magnitude_scale=1.0,
    ):
        super().__init__()

        # ---- Validation ----
        assert (
            len(num_blocks) == len(num_heads) == len(channels)
        ), "num_blocks, num_heads, and channels must have the same length"
        # (Relaxed) allow L == 1 now
        self.L = len(channels)
        self.channels = channels
        self.time_cond = time_cond
        self.label_cond = label_cond
        self.num_classes = num_classes
        self.timestep_scale = timestep_scale
        self.label_class_scale = label_class_scale
        self.ms_refinement = ms_refinement
        L = self.L

        self.emb_channels = channels[0] * time_embed_scale if time_cond else None

        # Shallow embedding at level 0
        self.embed_conv = nn.Conv2d(in_channel, channels[0], kernel_size=3, padding=1, bias=False)

        # Encoders: level 0..L-1
        self.encoders = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    *[
                        TransformerBlock(
                            channels[i],
                            num_heads[i],
                            expansion_factor,
                            num_experts,
                            top_k,
                            norm_type=norm_type,
                            post_norm=post_norm,
                            emb_channels=self.emb_channels,
                        )
                        for _ in range(num_blocks[i])
                    ]
                )
                for i in range(L)
            ]
        )

        # Down path: L-1 transitions
        self.downs = nn.ModuleList([DownSample(channels[i]) for i in range(L - 1)])

        # Up path: L-1 transitions (index i corresponds to going from level i+1 -> i)
        self.ups = nn.ModuleList([UpSample(channels[i + 1]) for i in range(L - 1)])

        # Reduce 1x1 after concatenation at each decoder stage i (levels i = L-2 .. 0)
        self.reduces = nn.ModuleList(
            [nn.Conv2d(2 * channels[i], channels[i], kernel_size=1, bias=False) for i in range(L - 1)]
        )

        # Decoders: one per stage for levels i = 0..L-2
        self.decoders = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    *[
                        TransformerBlock(
                            channels[i],
                            num_heads[i],
                            expansion_factor,
                            num_experts,
                            top_k,
                            drop_path=drop_path,
                            norm_type=norm_type,
                            post_norm=post_norm,
                            emb_channels=self.emb_channels,
                        )
                        for _ in range(num_blocks[i])
                    ]
                )
                for i in range(L - 1)
            ]
        )
        self.refinement = TimestepEmbedSequential(
            *[
                TransformerBlock(
                    channels[0],
                    num_heads[0],
                    expansion_factor,
                    num_experts,
                    top_k,
                    drop_path=drop_path,
                    norm_type=norm_type,
                    post_norm=post_norm,
                    emb_channels=self.emb_channels,
                )
                for _ in range(num_refinement)
            ]
        )
        # Final conv
        self.output = nn.Conv2d(channels[0], out_channel, kernel_size=3, padding=1, bias=False)

        # For padding to be divisible by 2^(L-1)
        self.pad_factor = 2 ** (L - 1)

        if self.time_cond:
            self.time_embed = nn.Sequential(
                nn.Linear(self.channels[0], self.emb_channels),
                nn.SiLU(),
                nn.Linear(self.emb_channels, self.emb_channels),
            )

        if self.label_cond:
            # self.label_emb = LabelEmbedder(self.num_classes, self.emb_channels, class_dropout_prob, labels_embed_magnitude_scale)
            self.label_emb = nn.Sequential(
                nn.Linear(self.channels[0], self.emb_channels),
                nn.SiLU(),
                nn.Linear(self.emb_channels, self.emb_channels),
            )

    def forward(self, x, cas_skips=None, timestep=None, label=None):
        L = self.L

        # Enforce cas_skips contract if provided
        if cas_skips is not None:
            if L >= 2:
                assert len(cas_skips) == (L - 1), f"cas_skips must have length {L-1} if L >= 2, got {len(cas_skips)}"
            else:
                assert len(cas_skips) == 1, f"cas_skips must have length 1 if L = 1, got {len(cas_skips)}"

        x, padding_sizes = divisible_pad_t(x, k=self.pad_factor)

        if timestep is not None and self.time_cond:
            timesteps = torch.tensor(float(timestep) * self.timestep_scale, device=x.device, dtype=x.dtype).expand(
                x.shape[0]
            )
            t_emb = timestep_embedding(timesteps, self.channels[0], repeat_only=False)
            emb = self.time_embed(t_emb)
        else:
            emb = None

        if label is not None and self.label_cond:
            y = torch.tensor(float(label) * self.label_class_scale, device=x.device, dtype=x.dtype).expand(x.shape[0])
            y_emb = timestep_embedding(y, self.channels[0], repeat_only=False)
            emb = emb + self.label_emb(y_emb) if emb is not None else self.label_emb(y_emb)

        # ---- Encoder ----
        feats = [None] * L  # store encoder outputs at each level
        x0 = self.embed_conv(x)
        feats[0] = self.encoders[0](x0, emb=emb)

        for i in range(1, L):
            down_in = self.downs[i - 1](feats[i - 1])
            if cas_skips is not None:
                # cas_skips length is L-1, where cas_skips[i-1] matches level i
                down_in = down_in + cas_skips[i - 1]
            feats[i] = self.encoders[i](down_in, emb=emb)

        # ---- Decoder ----
        if L >= 2:
            x_dec = feats[-1]  # start from bottom level (L-1)
            dec_outs = [None] * (L - 1)  # per-level decoder outputs for i=0..L-2

            # iterate levels from L-2 down to 0
            for i in reversed(range(L - 1)):
                up = self.ups[i](x_dec)  # upsample from level i+1 -> i
                cat = torch.cat([up, feats[i]], dim=1)
                red = self.reduces[i](cat)  # project 2*C_i -> C_i
                x_dec = self.decoders[i](red, emb=emb)  # decode at level i
                dec_outs[i] = x_dec  # store decoder output aligned with level i

            ref_in = dec_outs[0]  # refinement at top level (level 0), unchanged
        else:
            # L == 1: no decoder stages; refine the single encoder output
            dec_outs = []
            ref_in = feats[0] + cas_skips[0] if cas_skips is not None else feats[0]

        # ---- Refinement + Output ----
        fr = self.refinement(ref_in, emb=emb)
        out = self.output(fr)

        out = inverse_divisible_pad_t(out, padding_sizes)

        # ---- Cascade outputs (generalized) ----
        # Return: [dec_out(level=1), ..., dec_out(level=L-2), enc_out(level=L-1)]
        cas_outs = []
        if L >= 2:
            for i in range(1, L - 1):
                cas_outs.append(dec_outs[i])  # decoder outputs for levels 1..L-2
            cas_outs.append(feats[-1])  # deepest encoder output at level L-1
        else:
            cas_outs.append(fr)  # L == 1: empty cascade outputs

        return out, tuple(cas_outs)


class HybridRestormer(nn.Module):
    def __init__(
        self,
        in_channel=2,
        out_channel=2,
        num_blocks=[4, 6, 6, 8],
        num_heads=[1, 2, 4, 8],
        num_spatial_heads=[1, 2, 3, 4],
        channels=[48, 96, 192, 384],
        num_refinement=4,
        expansion_factor=2.66,
        num_experts=None,
        top_k=1,
        drop_path=0.0,
        norm_type=None,
        post_norm=False,
    ):
        super(HybridRestormer, self).__init__()
        self.embed_conv = nn.Conv2d(in_channel, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        HybridTransformerBlock(
                            num_ch,
                            num_ah,
                            num_sh,
                            expansion_factor,
                            num_experts,
                            top_k,
                            norm_type=norm_type,
                            post_norm=post_norm,
                        )
                        for _ in range(num_tb)
                    ]
                )
                for num_tb, num_ah, num_sh, num_ch in zip(num_blocks, num_heads, num_spatial_heads, channels)
            ]
        )

        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])
        self.reduces = nn.ModuleList(
            [
                nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                for i in reversed(range(2, len(channels)))
            ]
        )

        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        HybridTransformerBlock(
                            channels[2],
                            num_heads[2],
                            num_spatial_heads[2],
                            expansion_factor,
                            num_experts,
                            top_k,
                            drop_path,
                            norm_type=norm_type,
                            post_norm=post_norm,
                        )
                        for _ in range(num_blocks[2])
                    ]
                )
            ]
        )
        self.decoders.append(
            nn.Sequential(
                *[
                    HybridTransformerBlock(
                        channels[1],
                        num_heads[1],
                        num_spatial_heads[2],
                        expansion_factor,
                        num_experts,
                        top_k,
                        drop_path,
                        norm_type=norm_type,
                        post_norm=post_norm,
                    )
                    for _ in range(num_blocks[1])
                ]
            )
        )
        self.decoders.append(
            nn.Sequential(
                *[
                    HybridTransformerBlock(
                        channels[1],
                        num_heads[0],
                        num_spatial_heads[2],
                        expansion_factor,
                        num_experts,
                        top_k,
                        drop_path,
                        norm_type=norm_type,
                        post_norm=post_norm,
                    )
                    for _ in range(num_blocks[0])
                ]
            )
        )

        self.refinement = nn.Sequential(
            *[
                HybridTransformerBlock(
                    channels[1],
                    num_heads[0],
                    num_spatial_heads[2],
                    expansion_factor,
                    num_experts,
                    top_k,
                    norm_type=norm_type,
                    post_norm=post_norm,
                )
                for _ in range(num_refinement)
            ]
        )

        self.output = nn.Conv2d(channels[1], out_channel, kernel_size=3, padding=1, bias=False)
        self.pad_factor = 2 ** (len(channels) - 1) * 8

    def forward(self, x, cas_skips=None):
        x, padding_sizes = divisible_pad_t(x, k=self.pad_factor)
        fo = self.embed_conv(x)
        out_enc1 = self.encoders[0](fo)
        out_enc2 = (
            self.encoders[1](self.downs[0](out_enc1) + cas_skips[0])
            if cas_skips
            else self.encoders[1](self.downs[0](out_enc1))
        )
        out_enc3 = (
            self.encoders[2](self.downs[1](out_enc2) + cas_skips[1])
            if cas_skips
            else self.encoders[2](self.downs[1](out_enc2))
        )
        out_enc4 = (
            self.encoders[3](self.downs[2](out_enc3) + cas_skips[2])
            if cas_skips
            else self.encoders[3](self.downs[2](out_enc3))
        )

        out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc4), out_enc3], dim=1)))
        out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3), out_enc2], dim=1)))
        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), out_enc1], dim=1))
        fr = self.refinement(fd)
        out = self.output(fr)  # + x
        out = inverse_divisible_pad_t(out, padding_sizes)
        return out, (out_dec2, out_dec3, out_enc4)


class restormer_mri(nn.Module):
    def __init__(self, args, use_acs_region=None, last_cascade=False, first_cascade=False):
        super().__init__()

        self.args = args
        self.use_csm = args.use_csm
        self.use_acs_region = use_acs_region if use_acs_region is not None else args.use_acs_region
        self.num_frames = int(args.num_frames)
        self.num_coils = 10
        self.num_reduced_coils = int(args.num_reduced_coils) if hasattr(args, "num_reduced_coils") else 1

        self.mask_types = (
            [m.split("_")[-1] for m in args.train_mask_types_for_def_model]
            if hasattr(args, "train_mask_types_for_def_model")
            else [m.split("_")[-1] for m in args.train_mask_types]
        )
        self.acq_types = (
            args.acq_types
            if hasattr(args, "acq_types")
            else [
                "BlackBlood",
                "Cine",
                "Flow2d",
                "LGE",
                "Mapping",
                "Perfusion",
                "T1rho",
                "T1w",
                "T2w",
                "Aorta",
                "Tagging",
            ]
        )
        self.acc_factors = (
            [int(m) for m in args.accelerations_for_def_model]
            if hasattr(args, "accelerations_for_def_model")
            else [int(m) for m in args.accelerations]
        )

        self.drop_path = args.drop_path if hasattr(args, "drop_path") else 0.0
        self.last_cascade = last_cascade
        self.first_cascade = first_cascade
        self.adaptive_temporal_dc = args.adaptive_temporal_dc if hasattr(args, "adaptive_temporal_dc") else False
        self.gd = args.gd if hasattr(args, "gd") else False
        self.num_experts = args.num_experts if hasattr(args, "num_experts") else None
        self.top_k = args.top_k if hasattr(args, "top_k") else 1
        self.num_blocks = args.num_blocks if hasattr(args, "num_blocks") else [4, 6, 6, 8]
        self.num_heads = args.num_heads if hasattr(args, "num_heads") else [1, 2, 4, 8]
        self.num_refinement = args.num_refinement if hasattr(args, "num_refinement") else 4
        self.channels = args.channels if hasattr(args, "channels") else [48, 96, 192, 384]
        self.mlp_ratio = args.mlp_ratio if hasattr(args, "mlp_ratio") else 4
        self.norm_type = args.norm_type if hasattr(args, "norm_type") else None
        self.hybrid_attn = args.hybrid_attn if hasattr(args, "hybrid_attn") else False
        self.post_norm = args.post_norm if hasattr(args, "post_norm") else False
        self.mask_specific_dc_weight_map = (
            args.mask_specific_dc_weight_map if hasattr(args, "mask_specific_dc_weight_map") else False
        )
        self.use_single_csm = args.use_single_csm if hasattr(args, "use_single_csm") else False
        self.ms_refinement = args.ms_refinement if hasattr(args, "ms_refinement") else False
        self.time_cond = args.time_cond if hasattr(args, "time_cond") else False
        self.label_cond = args.label_cond if hasattr(args, "label_cond") else False
        self.num_classes = (
            args.num_classes
            if hasattr(args, "num_classes")
            else len(self.mask_types) * len(self.acq_types) * len(self.acc_factors)
        )
        labels_embed_magnitude_scale = (
            args.labels_embed_magnitude_scale if hasattr(args, "labels_embed_magnitude_scale") else 1.0
        )

        if self.label_cond and not hasattr(args, "acq_types"):
            print(
                f"label_cond enabled but without acq types specified in config file, using default acq types: {self.acq_types}"
            )

        restormer = HybridRestormer if self.hybrid_attn else Restormer

        self.pad_factor = 2**2
        if self.use_csm:
            if (self.first_cascade and self.use_single_csm) or not self.use_single_csm:
                self.coil_sensitivity_model = CoilSensitivityModel_DCAE(
                    spatial_dims=2,
                    features=(12, 24, 48, 96, 192),
                    pad_factor=self.pad_factor,
                )
            if self.args.pretrained_csm is not None:
                self.load_csm_model()
            self.recon_model = restormer(
                in_channel=self.num_frames * self.num_reduced_coils * 2,
                out_channel=self.num_frames * self.num_reduced_coils * 2,
                num_experts=self.num_experts,
                top_k=self.top_k,
                expansion_factor=self.mlp_ratio,
                num_blocks=self.num_blocks,
                num_heads=self.num_heads,
                num_refinement=self.num_refinement,
                channels=self.channels,
                drop_path=self.drop_path,
                norm_type=self.norm_type,
                post_norm=self.post_norm,
                ms_refinement=self.ms_refinement,
                time_cond=self.time_cond,
                label_cond=self.label_cond,
                num_classes=self.num_classes,
                labels_embed_magnitude_scale=labels_embed_magnitude_scale,
            )
        else:
            self.recon_model = restormer(
                in_channel=self.num_frames * self.num_coils * 2,
                out_channel=self.num_frames * self.num_coils * 2,
                num_experts=self.num_experts,
                top_k=self.top_k,
                expansion_factor=self.mlp_ratio,
                num_blocks=self.num_blocks,
                num_heads=self.num_heads,
                num_refinement=self.num_refinement,
                channels=self.channels,
                drop_path=self.drop_path,
                norm_type=self.norm_type,
                post_norm=self.post_norm,
                ms_refinement=self.ms_refinement,
                time_cond=self.time_cond,
                label_cond=self.label_cond,
                num_classes=self.num_classes,
                labels_embed_magnitude_scale=labels_embed_magnitude_scale,
            )
        self.use_dc_weight_map = args.use_dc_weight_map
        if self.use_dc_weight_map:
            dcwm_h, dcwm_w = (328, 806) if args.dataset.lower() == "cmrxrecon" else (768, 768)
            if self.mask_specific_dc_weight_map:
                self.dc_weight_map = nn.Parameter(
                    torch.ones(len(self.mask_types), self.num_frames, 1, dcwm_h, dcwm_w, 1)
                )  # 640 x 640 should be large enough
            else:
                self.dc_weight_map = nn.Parameter(
                    torch.ones(self.num_frames, 1, dcwm_h, dcwm_w, 1)
                )  # 640 x 640 should be large enough
        else:
            self.dc_weight = nn.Parameter(torch.ones(1))

        if self.args.pretrained_recon is not None:
            self.recon_model = self.load_recon_model(self.recon_model)

    def load_recon_model(self, recon_model):
        try:
            loaded_state_dict = torch.load(self.args.pretrained_recon, map_location="cpu", weights_only=True)
        except BaseException:
            loaded_state_dict = torch.load(self.args.pretrained_recon, map_location="cpu", weights_only=False)[
                "net_state_dict"
            ]
        current_model_dict = recon_model.state_dict()
        new_state_dict = {}
        loaded_keys = []
        for k in current_model_dict.keys():
            if k in loaded_state_dict:
                if loaded_state_dict[k].size() == current_model_dict[k].size():
                    new_state_dict[k] = loaded_state_dict[k]
                    loaded_keys.append(k)
            else:
                new_state_dict[k] = current_model_dict[k]

        recon_model.load_state_dict(new_state_dict)
        print(f"Loaded pretrained Recon model from {self.args.pretrained_recon}")
        print(
            f"Loaded {len(loaded_keys)} keys from pretrained Recon model with {len(loaded_state_dict)} keys \
            from {self.args.pretrained_recon}."
        )
        return recon_model

    def load_csm_model(self):
        try:
            csm_state_dict = torch.load(self.args.pretrained_csm, map_location="cpu", weights_only=True)[
                "net_state_dict"
            ]
        except BaseException:
            csm_state_dict = torch.load(self.args.pretrained_csm, map_location="cpu", weights_only=False)[
                "net_state_dict"
            ]
        _, updated_keys, unchanged_keys = copy_model_state(self, csm_state_dict)
        print(f"Loaded pretrained CSM model from {self.args.pretrained_csm}")

    def hard_dc(self, x: torch.Tensor, ref_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        kspace_pred = fftn_centered(x, spatial_dims=2)
        ref_kspace = fftn_centered(ref_image, spatial_dims=2)

        kspace_pred = ~mask * kspace_pred + mask * ref_kspace
        return ifftn_centered(kspace_pred, spatial_dims=2)

    def soft_dc(
        self,
        x: torch.Tensor,
        ref_image: torch.Tensor,
        mask: torch.Tensor,
        temporal_weight: torch.Tensor = None,
        mask_type: str = None,
    ) -> torch.Tensor:
        kspace_pred = fftn_centered(x, spatial_dims=2)
        ref_kspace = fftn_centered(ref_image, spatial_dims=2)

        if self.use_dc_weight_map:
            if self.mask_specific_dc_weight_map and mask_type is not None:
                dc_weight_map = self.dc_weight_map[
                    next(
                        (i for i, sub in enumerate(self.mask_types) if sub in mask_type.lower()),
                        -1,
                    )
                ]
            else:
                dc_weight_map = self.dc_weight_map

            _, _, H, W, _ = kspace_pred.shape
            _, _, H_map, W_map, _ = dc_weight_map.shape

            if H_map < H or W_map < W:
                pad_h = max(H - H_map, 0)
                pad_w = max(W - W_map, 0)

                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left

                weight_map_perm = dc_weight_map.permute(0, 1, 4, 2, 3)

                # Pad (W_left, W_right, H_top, H_bottom)
                weight_map_perm = F.pad(
                    weight_map_perm,
                    (pad_left, pad_right, pad_top, pad_bottom, 0, 0),
                    mode="replicate",
                )

                dc_weight_map = weight_map_perm.permute(0, 1, 3, 4, 2)
                _, _, H_map, W_map, _ = dc_weight_map.shape

            top = (H_map - H) // 2
            left = (W_map - W) // 2
            if temporal_weight:
                weight = (
                    dc_weight_map[None, :, :, top : top + H, left : left + W, :] * temporal_weight[:, : self.num_frames]
                )
                weight = weight.view(weight.shape[0] * weight.shape[1], *weight.shape[2:])
            else:
                weight = dc_weight_map[:, :, top : top + H, left : left + W, :].repeat(
                    int(mask.shape[0] / self.num_frames), 1, 1, 1, 1
                )
        else:
            weight = self.dc_weight

        kspace_pred = ~mask * kspace_pred + mask * ((1 - weight) * kspace_pred + weight * ref_kspace)
        return ifftn_centered(kspace_pred, spatial_dims=2)

    def forward(
        self,
        x: torch.Tensor,
        ref_image: torch.Tensor,
        mask: torch.Tensor = None,
        cas_skips: torch.Tensor = None,
        mask_type: str = None,
        acc_factor: int = None,
        acq_type: str = None,
        sensitivity_maps: torch.Tensor = None,
        timestep: int = None,
    ) -> tuple[Tensor | Any, Any]:
        # x shape: (B,T,C,H,W,2) undersampled image
        # mask shape: (B,T,C,H,W,2) mask

        B, T, C, H, W, two = x.shape
        x = rearrange(x, "b t c h w two-> (b t) c h w two")
        ref_image = rearrange(ref_image, "b t c h w two-> (b t) c h w two")
        mask = rearrange(mask, "b t c h w two-> (b t) c h w two")

        skip = x.clone()

        if self.use_csm:
            if sensitivity_maps is None or not self.use_single_csm:
                if self.use_acs_region and mask is not None:
                    x_acs = get_acs_image(x, mask)
                    if self.use_single_csm:
                        sensitivity_maps = self.coil_sensitivity_model(x_acs)  # shape (B,C,H,W,2)
                    else:
                        sensitivity_maps = checkpoint(self.coil_sensitivity_model, x_acs, use_reentrant=False)
                else:
                    if self.use_single_csm:
                        sensitivity_maps = self.coil_sensitivity_model(x)  # shape (B,C,H,W,2)
                    else:
                        sensitivity_maps = checkpoint(self.coil_sensitivity_model, x, use_reentrant=False)
            x = sensitivity_map_reduce(x, sensitivity_maps, k=self.num_reduced_coils)  # x will be of shape (B,1,H,W,2)

        x = rearrange(x, "(b t) c h w two -> b (t c two) h w", t=T, two=2)
        mask_idx = next((i for i, sub in enumerate(self.mask_types) if sub in mask_type.lower()), -1)
        acc_idx = next((i for i, sub in enumerate(self.acc_factors) if int(sub) == acc_factor), -1)
        acq_idx = next(
            (i for i, sub in enumerate(self.acq_types) if sub.lower() == acq_type.lower()),
            -1,
        )
        assert acq_idx != -1, f"acq type {acq_type} not found in {self.acq_types}"
        x, cas_skips = self.recon_model(
            x,
            cas_skips,
            timestep,
            mask_idx * len(self.mask_types) * len(self.acq_types) + acc_idx * len(self.acq_types) + acq_idx,
        )
        x = rearrange(x, "b (t c two) h w -> (b t) c h w two", t=T, two=2)

        if self.last_cascade:
            skip = rearrange(skip, "(b t) c h w two -> b t c h w two", t=T)[:, T // 2 : T // 2 + 1, :, :, :, :].expand(
                -1, T, -1, -1, -1, -1
            )
            skip = rearrange(skip, "b t c h w two-> (b t) c h w two")
            ref_image = rearrange(ref_image, "(b t) c h w two -> b t c h w two", t=T)[
                :, T // 2 : T // 2 + 1, :, :, :, :
            ].expand(-1, T, -1, -1, -1, -1)
            ref_image = rearrange(ref_image, "b t c h w two-> (b t) c h w two")
            mask = rearrange(mask, "(b t) c h w two -> b t c h w two", t=T)[:, T // 2 : T // 2 + 1, :, :, :, :].expand(
                -1, T, -1, -1, -1, -1
            )
            mask = rearrange(mask, "b t c h w two-> (b t) c h w two")
            if self.use_csm:
                sensitivity_maps = rearrange(sensitivity_maps, "(b t) c h w two -> b t c h w two", t=T)[
                    :, T // 2 : T // 2 + 1, :, :, :, :
                ].expand(-1, T, -1, -1, -1, -1)
                sensitivity_maps = rearrange(sensitivity_maps, "b t c h w two-> (b t) c h w two")

        if self.use_csm:
            x = sensitivity_map_expand(x, sensitivity_maps)  # x will be of shape (B,C,H,W,2)
        # x shape: (B,C,H,W,2)

        if not self.gd:
            x = skip + x
            if mask is not None:
                with torch.autocast("cuda", torch.bfloat16, enabled=False):
                    x = self.soft_dc(x.float(), ref_image, mask, mask_type=mask_type)  # (B,C,H,W,2)
        else:
            if mask is not None:
                with torch.autocast("cuda", torch.bfloat16, enabled=False):
                    skip = self.soft_dc(skip.float(), ref_image, mask, mask_type=mask_type)  # (B,C,H,W,2)
            x = skip + x

        x = rearrange(x, "(b t) c h w two -> b t c h w two", t=T)

        if self.last_cascade:
            x = torch.mean(x, dim=1, keepdim=True).expand(-1, T, -1, -1, -1, -1)

        return x, cas_skips, sensitivity_maps
