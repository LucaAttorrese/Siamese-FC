"""SiamMask network ("siammask_sharp" Custom model, VOT weights), ported to current PyTorch.

Port of SiamMask (https://github.com/foolwood/SiamMask, commit a2b3ce3), MIT License,
Copyright (c) 2019 Qiang Wang. Sources: experiments/siammask_sharp/{custom.py,resnet.py},
models/{siammask_sharp.py,rpn.py}. Feasibility study only (proposal P7 in PLAN.md, step 0):
not used by the runtime.

The module tree and parameter names are the same as in the original, so the published
``SiamMask_VOT.pth`` state dict loads without renaming (after removing the ``module.`` prefix).
Training-only code is left out. Kept on purpose, as in the original:
- ``Bottleneck`` padding ``2 - stride`` (0 for the stride-2 block of layer2);
- ``F.upsample`` default mode, i.e. nearest, in the refine module;
- the 7x7 centre crop of the template features in ``ResDownS``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        padding = 2 - stride
        assert stride == 1 or dilation == 1
        if dilation > 1:
            padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + residual)


class ResNet(nn.Module):
    """ResNet-50 up to layer3 (layer3 dilated, stride 8 overall)."""

    def __init__(self, layers=(3, 4, 6)):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=1, dilation=2)

    def _make_layer(self, planes, blocks, stride=1, dilation=1):
        downsample = None
        dd = dilation
        if stride != 1 or self.inplanes != planes * 4:
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * 4))
            else:
                if dilation > 1:
                    dd, padding = dilation // 2, dilation // 2
                else:
                    dd, padding = 1, 0
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * 4, kernel_size=3, stride=stride, bias=False, padding=padding, dilation=dd),
                    nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dd)]
        self.inplanes = planes * 4
        layers += [Bottleneck(self.inplanes, planes, dilation=dilation) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        p0 = self.relu(self.bn1(self.conv1(x)))
        p1 = self.layer1(self.maxpool(p0))
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)
        return p0, p1, p2, p3


class ResDownS(nn.Module):
    def __init__(self, inplane, outplane):
        super().__init__()
        self.downsample = nn.Sequential(nn.Conv2d(inplane, outplane, kernel_size=1, bias=False), nn.BatchNorm2d(outplane))

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:  # template: 15x15 -> centre 7x7
            x = x[:, :, 4:-4, 4:-4]
        return x


class ResDown(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = ResNet()
        self.downsample = ResDownS(1024, 256)

    def forward(self, x):
        return self.downsample(self.features(x)[-1])

    def forward_all(self, x):
        output = self.features(x)
        return output, self.downsample(output[-1])


def conv2d_dw_group(x, kernel):
    batch, channel = kernel.shape[:2]
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch * channel)
    return out.view(batch, channel, out.size(2), out.size(3))


class DepthCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super().__init__()
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True))
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1))

    def forward_corr(self, kernel, search):
        return conv2d_dw_group(self.conv_search(search), self.conv_kernel(kernel))

    def forward(self, kernel, search):
        return self.head(self.forward_corr(kernel, search))


class UP(nn.Module):
    def __init__(self, anchor_num=5, feature_in=256, feature_out=256):
        super().__init__()
        self.cls = DepthCorr(feature_in, feature_out, 2 * anchor_num)
        self.loc = DepthCorr(feature_in, feature_out, 4 * anchor_num)

    def forward(self, z_f, x_f):
        return self.cls(z_f, x_f), self.loc(z_f, x_f)


class MaskCorr(nn.Module):
    def __init__(self, o_sz=63):
        super().__init__()
        self.mask = DepthCorr(256, 256, o_sz ** 2)


class Refine(nn.Module):
    def __init__(self):
        super().__init__()
        self.v0 = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 4, 3, padding=1), nn.ReLU())
        self.v1 = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 16, 3, padding=1), nn.ReLU())
        self.v2 = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 32, 3, padding=1), nn.ReLU())
        self.h2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.h1 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.h0 = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 4, 3, padding=1), nn.ReLU())
        self.deconv = nn.ConvTranspose2d(256, 32, 15, 15)
        self.post0 = nn.Conv2d(32, 16, 3, padding=1)
        self.post1 = nn.Conv2d(16, 4, 3, padding=1)
        self.post2 = nn.Conv2d(4, 1, 3, padding=1)

    def forward(self, f, corr_feature, pos):
        """Test-time path of the original (``test=True``): one 127x127 mask logit map for the
        score-map position ``pos`` = (y, x)."""
        y, x = int(pos[0]), int(pos[1])
        p0 = F.pad(f[0], [16, 16, 16, 16])[:, :, 4 * y:4 * y + 61, 4 * x:4 * x + 61]
        p1 = F.pad(f[1], [8, 8, 8, 8])[:, :, 2 * y:2 * y + 31, 2 * x:2 * x + 31]
        p2 = F.pad(f[2], [4, 4, 4, 4])[:, :, y:y + 15, x:x + 15]
        p3 = corr_feature[:, :, y, x].reshape(-1, 256, 1, 1)
        out = self.deconv(p3)
        out = self.post0(F.interpolate(self.h2(out) + self.v2(p2), size=(31, 31)))  # nearest, as F.upsample
        out = self.post1(F.interpolate(self.h1(out) + self.v1(p1), size=(61, 61)))
        out = self.post2(F.interpolate(self.h0(out) + self.v0(p0), size=(127, 127)))
        return out.view(-1, 127 * 127)


class SiamMaskNet(nn.Module):
    """``Custom`` model of experiments/siammask_sharp (anchor_num 5)."""

    def __init__(self, anchor_num: int = 5):
        super().__init__()
        self.features = ResDown()
        self.rpn_model = UP(anchor_num=anchor_num)
        self.mask_model = MaskCorr()
        self.refine_model = Refine()
        self.zf = None

    @torch.no_grad()
    def template(self, z):
        self.zf = self.features(z)

    @torch.no_grad()
    def track(self, x):
        return self.rpn_model(self.zf, self.features(x))

    @torch.no_grad()
    def track_mask(self, x):
        """Score, box deltas, and the state the refine module needs. The coarse mask head
        (63x63 per position) is not computed: with the refine module, as in the VOT setting,
        the original computes it and throws it away."""
        feature, search = self.features.forward_all(x)
        cls, loc = self.rpn_model(self.zf, search)
        corr = self.mask_model.mask.forward_corr(self.zf, search)
        return cls, loc, (feature, corr)

    @torch.no_grad()
    def track_refine(self, mask_state, pos):
        feature, corr = mask_state
        return self.refine_model(feature, corr, pos)


def _numpy_safe_globals() -> list:
    """The 2019 checkpoint (legacy pickle format) stores a few numpy scalars next to the state
    dict (checked with pickletools: the only globals are torch storages/_rebuild_tensor_v2,
    OrderedDict, numpy dtype, numpy scalar and _codecs.encode). These are data types, so they
    are allow-listed for the ``weights_only`` unpickler instead of loading with full pickle."""
    import numpy as np
    try:
        from numpy._core.multiarray import scalar
    except ImportError:  # numpy < 2
        from numpy.core.multiarray import scalar
    out = [(scalar, "numpy.core.multiarray.scalar"), np.dtype]
    out += [type(np.dtype(t)) for t in ("float64", "float32", "int64", "int32")]  # numpy >= 1.25 dtype classes
    return out


def load_siammask(path: str, device: str | torch.device = "cpu") -> tuple[SiamMaskNet, dict]:
    """Load ``SiamMask_VOT.pth`` with ``weights_only=True`` (no code from the file is run).
    Returns the network in eval mode and a report of the key matching."""
    with torch.serialization.safe_globals(_numpy_safe_globals()):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    net = SiamMaskNet()
    model_keys = set(net.state_dict().keys())
    missing = sorted(model_keys - set(sd))
    unexpected = sorted(set(sd) - model_keys)
    bad_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if bad_missing:
        raise ValueError(f"checkpoint lacks {len(bad_missing)} parameters, e.g. {bad_missing[:5]}")
    net.load_state_dict({k: v for k, v in sd.items() if k in model_keys}, strict=False)
    net.eval().to(device)
    return net, {"missing": missing, "unexpected": unexpected, "loaded": len(model_keys & set(sd))}
