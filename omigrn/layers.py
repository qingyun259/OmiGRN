import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    return (d * (k - 1)) // 2 if p is None else p


def round_ch(val, div):
    return max(1, int((val + div / 2) // div * div))


def _to_3d(x):
    if x.dim() == 2:
        return x.unsqueeze(1), True
    if x.dim() == 3:
        return x, False
    raise ValueError(f"Expected 2D or 3D tensor, got {tuple(x.shape)}")


def _from_3d(x, squeezed, flatten):
    if not squeezed:
        return x
    if x.size(1) == 1:
        return x[:, 0, :]
    return x.flatten(1) if flatten else x


class CBS(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv1d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm1d(c2)
        if act is True:
            self.act = nn.SiLU()
        elif act is False:
            self.act = nn.Identity()
        else:
            self.act = act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=3, e=0.5):
        super().__init__()
        c_ = max(1, int(c2 * e))
        self.cv1 = CBS(c1, c_, 1, 1)
        self.cv2 = CBS(c_, c2, k, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class CSPBlock(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, flatten=True):
        super().__init__()
        self.c = max(1, int(c2 * e))
        self.flatten = flatten
        self.cv1 = CBS(c1, 2 * self.c, 1, 1)
        self.cv2 = CBS((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=3, e=1.0) for _ in range(n))

    def forward(self, x):
        x, squeezed = _to_3d(x)
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(blk(y[-1]) for blk in self.m)
        return _from_3d(self.cv2(torch.cat(y, 1)), squeezed, self.flatten)


class CAA(nn.Module):
    def __init__(self, ch, k=11):
        super().__init__()
        self.avg_pool = nn.AvgPool1d(7, 1, 3)
        self.conv1 = CBS(ch, ch, 1, 1)
        self.dw_conv = nn.Conv1d(ch, ch, kernel_size=k, padding=autopad(k), groups=ch, bias=False)
        self.conv2 = CBS(ch, ch, 1, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.conv2(self.dw_conv(self.conv1(self.avg_pool(x)))))


class PKIBlock(nn.Module):
    def __init__(self, c1, c2, kernels=(3, 5, 7, 9, 11), expansion=1.0, use_caa=True, caa_k=11, shortcut=True, flatten=True):
        super().__init__()
        hid = round_ch(int(c2 * expansion), 8)
        self.flatten = flatten
        self.pre_conv = CBS(c1, hid)
        self.dw_conv = nn.ModuleList(
            nn.Conv1d(hid, hid, kernel_size=k, padding=autopad(k), groups=hid, bias=False)
            for k in kernels
        )
        self.pw_conv = CBS(hid, hid)
        self.post_conv = CBS(hid, c2)
        self.caa_factor = CAA(hid, caa_k) if use_caa else None
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        x, squeezed = _to_3d(x)
        x = self.pre_conv(x)
        y = x
        x = torch.sum(torch.stack([x] + [conv(x) for conv in self.dw_conv], dim=0), dim=0)
        x = self.pw_conv(x)
        if self.caa_factor is not None:
            y = self.caa_factor(y)
        x = x + x * y if self.shortcut else x * y
        return _from_3d(self.post_conv(x), squeezed, self.flatten)


class MGCM(CSPBlock):
    def __init__(self, c1=1, c2=1, n=1, kernels=(3, 5, 7, 9, 11), expansion=1.0, use_caa=True, caa_k=11, shortcut=True, g=1, e=0.5, flatten=True):
        super().__init__(c1, c2, n, True, g, e, flatten=flatten)
        self.m = nn.ModuleList(
            PKIBlock(self.c, self.c, kernels, expansion, use_caa, caa_k, shortcut, flatten=False)
            for _ in range(n)
        )
