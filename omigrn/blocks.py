"""
Building blocks for the OmiGRN genomic branch — the multi-scale genomic context
mixer (MGCM).

Following the paper, the ``MGCM`` (used by ``ModalityEncoder`` in ``model.py``)
adopts a Cross-Stage-Partial (CSP) backbone (``CSPLayer``) that wraps cascaded
Poly-Kernel-Inception (``PKIModule``) units. Each PKI unit combines multi-scale
1-D depthwise convolutions with a Context-Anchor-Attention (``CAA``) gate. The
basic Conv1d -> BatchNorm -> SiLU operator is the ``CBS`` unit.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn


def autopad(k: int, p: int | None = None, d: int = 1) -> int:
    """Compute padding for 'same' shape 1D convolutions."""
    if p is not None:
        return p
    return (d * (k - 1)) // 2


def make_divisible(value: int, divisor: int) -> int:
    """Round channel counts to be divisible by a given divisor."""
    rounded = int((value + divisor / 2) // divisor * divisor)
    return max(1, rounded)


class CBS(nn.Module):
    """CBS unit: 1-D convolution -> BatchNorm -> SiLU activation."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm1d(c2)
        if act is True:
            self.act = nn.SiLU()
        elif act is False:
            self.act = nn.Identity()
        else:
            self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """1D bottleneck block with optional shortcut (CSP internal block)."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: int = 3,
        e: float = 0.5,
    ) -> None:
        super().__init__()
        c_ = max(1, int(c2 * e))
        self.cv1 = CBS(c1, c_, 1, 1)
        self.cv2 = CBS(c_, c2, k, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


def _ensure_3d(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if x.dim() == 2:
        return x.unsqueeze(1), True
    if x.dim() == 3:
        return x, False
    raise ValueError(f"Expected 2D or 3D input, got shape {tuple(x.shape)}")


def _restore_shape(
    x: torch.Tensor, squeezed: bool, flatten_output: bool
) -> torch.Tensor:
    if not squeezed:
        return x
    if x.size(1) == 1:
        return x[:, 0, :]
    return x.flatten(1) if flatten_output else x


class CSPLayer(nn.Module):
    """Cross-Stage-Partial (CSP) backbone for 1-D inputs.

    Splits the feature map into a shallow bypass and a main branch processed by
    ``n`` cascaded blocks, then concatenates and fuses them (the CSP design of
    the MGCM). Accepts input as (batch, length) or (batch, channels, length).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        flatten_output: bool = True,
    ) -> None:
        """Initialize the 1-D CSP layer."""
        super().__init__()
        self.c = max(1, int(c2 * e))  # hidden channels
        self.flatten_output = flatten_output
        self.cv1 = CBS(c1, 2 * self.c, 1, 1)
        self.cv2 = CBS((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=3, e=1.0) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP layer."""
        x, squeezed = _ensure_3d(x)
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return _restore_shape(out, squeezed, self.flatten_output)

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        x, squeezed = _ensure_3d(x)
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return _restore_shape(out, squeezed, self.flatten_output)


class CAA(nn.Module):
    """Context-Anchor-Attention (CAA) gate for 1-D signals."""

    def __init__(self, ch: int, kernel_size: int = 11) -> None:
        super().__init__()

        self.avg_pool = nn.AvgPool1d(7, 1, 3)
        self.conv1 = CBS(ch, ch, 1, 1)
        self.dw_conv = nn.Conv1d(
            ch, ch, kernel_size=kernel_size, padding=autopad(kernel_size), groups=ch, bias=False
        )
        self.conv2 = CBS(ch, ch, 1, 1)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_factor = self.act(self.conv2(self.dw_conv(self.conv1(self.avg_pool(x)))))
        return attn_factor


class PKIModule(nn.Module):
    """Poly-Kernel-Inception (PKI) unit adapted for 1-D signals.

    Aggregates multi-scale local context via parallel depthwise convolutions and
    refines it with a Context-Anchor-Attention (``CAA``) gate. Accepts input as
    (batch, length) or (batch, channels, length).
    """

    def __init__(
        self,
        inc: int,
        ouc: int,
        kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
        expansion: float = 1.0,
        with_caa: bool = True,
        caa_kernel_size: int = 11,
        add_identity: bool = True,
        flatten_output: bool = True,
    ) -> None:
        super().__init__()
        hidc = make_divisible(int(ouc * expansion), 8)
        self.flatten_output = flatten_output

        self.pre_conv = CBS(inc, hidc)
        self.dw_conv = nn.ModuleList(
            nn.Conv1d(
                hidc, hidc, kernel_size=k, padding=autopad(k), groups=hidc, bias=False
            )
            for k in kernel_sizes
        )
        self.pw_conv = CBS(hidc, hidc)
        self.post_conv = CBS(hidc, ouc)

        if with_caa:
            self.caa_factor = CAA(hidc, caa_kernel_size)
        else:
            self.caa_factor = None

        self.add_identity = add_identity and inc == ouc

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x, squeezed = _ensure_3d(x)
        x = self.pre_conv(x)

        y = x
        x = torch.sum(torch.stack([x] + [layer(x) for layer in self.dw_conv], dim=0), dim=0)
        x = self.pw_conv(x)

        if self.caa_factor is not None:
            y = self.caa_factor(y)

        if self.add_identity:
            y = x * y
            x = x + y

        else:
            x = x * y

        x = self.post_conv(x)

        return _restore_shape(x, squeezed, self.flatten_output)


class MGCM(CSPLayer):
    """Multi-scale Genomic Context Mixer (MGCM).

    A CSP backbone whose cascaded blocks are Poly-Kernel-Inception (``PKIModule``)
    units; this is the genomic-branch feature extractor described in the paper.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
        expansion: float = 1.0,
        with_caa: bool = True,
        caa_kernel_size: int = 11,
        add_identity: bool = True,
        g: int = 1,
        e: float = 0.5,
        flatten_output: bool = True,
    ) -> None:
        super().__init__(c1, c2, n, True, g, e, flatten_output=flatten_output)
        self.m = nn.ModuleList(
            PKIModule(
                self.c,
                self.c,
                kernel_sizes,
                expansion,
                with_caa,
                caa_kernel_size,
                add_identity,
                flatten_output=False,
            )
            for _ in range(n)
        )
