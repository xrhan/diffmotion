import torch
from torch import nn, einsum
from einops import rearrange, parse_shape, repeat
from typing import Optional, Tuple
import math
from rotary_embedding_torch.rotary_embedding_torch import rotate_half


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) from https://arxiv.org/abs/1910.07467.
    Reference: https://github.com/meta-llama/llama/blob/main/llama/model.py#L34-L77
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    

def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class RotaryEmbeddingND(nn.Module):
    """
    Minimal Axial RoPE generalized to N dimensions.
    """

    def __init__(
        self,
        dims: Tuple[int, ...],
        sizes: Tuple[int, ...],
        theta: float = 10000.0,
        flatten: bool = True,
    ):
        """
        Args:
            dims: the number of dimensions for each axis.
            sizes: the maximum length for each axis.
        """
        super().__init__()
        self.n_dims = len(dims)
        self.dims = dims
        self.theta = theta
        self.flatten = flatten

        Colon = slice(None)
        all_freqs = []
        for i, (dim, seq_len) in enumerate(zip(dims, sizes)):
            freqs = self.get_freqs(dim, seq_len)
            all_axis = [None] * len(dims)
            all_axis[i] = Colon
            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice].expand(*sizes, dim))
        all_freqs = torch.cat(all_freqs, dim=-1)
        if flatten:  # flatten all but the last dimension
            all_freqs = rearrange(all_freqs, "... d -> (...) d")
        self.register_buffer("freqs", all_freqs, persistent=False)

    def get_freqs(self, dim: int, seq_len: int) -> torch.Tensor:
        freqs = 1.0 / (
            self.theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
        )
        pos = torch.arange(seq_len, dtype=freqs.dtype)
        freqs = einsum("..., f -> ... f", pos, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        return freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: a [... x N x ... x D] if flatten=False, [... x (N x ...) x D] if flatten=True tensor of queries or keys.
        Returns:
            a tensor of rotated queries or keys. (same shape as x)
        """
        # slice the freqs to match the input shape
        seq_shape = x.shape[-2:-1] if self.flatten else x.shape[-self.n_dims - 1 : -1]
        slice_tuple = tuple(slice(0, seq_len) for seq_len in seq_shape)
        freqs = self.freqs[slice_tuple]
        return x * freqs.cos() + rotate_half(x) * freqs.sin()


class RotaryEmbedding1D(RotaryEmbeddingND):
    """
    RoPE1D for Time Series Transformer.
    Handles tensors of shape [B x T x C] or [B x (T x C)].
    """

    def __init__(
        self,
        dim: int,
        seq_len: int,
        theta: float = 10000.0,
        flatten: bool = True,
    ):
        super().__init__((dim,), (seq_len,), theta, flatten)


class RotaryEmbedding2D(RotaryEmbeddingND):
    """
    RoPE2D for Image Transformer.
    Handles tensors of shape [B x H x W x C] or [B x (H x W) x C].
    """

    def __init__(
        self,
        dim: int,
        sizes: Tuple[int, int],
        theta: float = 10000.0,
        flatten: bool = True,
    ):
        assert dim % 2 == 0, "RotaryEmbedding2D requires even dim"
        super().__init__((dim // 2,) * 2, sizes, theta, flatten)


class RotaryEmbedding3D(RotaryEmbeddingND):
    """
    RoPE3D for Video Transformer.
    Handles tensors of shape [B x T x H x W x C] or [B x (T x H x W) x C].
    """

    def __init__(
        self,
        dim: int,
        sizes: Tuple[int, int, int],
        theta: float = 10000.0,
        flatten: bool = True,
    ):
        assert dim % 2 == 0, "RotaryEmbedding3D requires even dim"
        dim //= 2

        # if dim is not divisible by 3,
        # split into 3 dimensions such that height and width have the same number of frequencies
        match dim % 3:
            case 0:
                dims = (dim // 3,) * 3
            case 1:
                dims = (dim // 3 + 1, dim // 3, dim // 3)
            case 2:
                dims = (dim // 3, dim // 3 + 1, dim // 3 + 1)

        super().__init__(tuple(d * 2 for d in dims), sizes, theta, flatten)
