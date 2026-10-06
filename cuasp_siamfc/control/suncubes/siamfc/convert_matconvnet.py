"""Convert a MatConvNet DagNN SiamFC/CFNet ``xcorr`` network to PyTorch weights.

Usage::

    python -m suncubes.siamfc.convert_matconvnet path/to/baseline-conv5_gray_e100.mat [out.pth]

Conversion rules (each one is checked by ``tests/test_siamfc_convert.py``):

- MatConvNet filters are ``H x W x Cin_per_group x Cout`` and are applied as a
  cross-correlation (no flip), like ``torch.nn.functional.conv2d``. They become
  ``Cout x Cin_per_group x H x W``. The group count is
  ``Cin_actual / Cin_per_group``, i.e. 2 for conv2/conv4/conv5 (AlexNet split).
- ``dagnn.BatchNorm`` in test mode computes ``y = m * (x - mu) / sigma + b``,
  where ``moments = [mu, sigma]`` and sigma is already ``sqrt(var + eps)``
  (a standard deviation, not a variance). The BN that follows each conv is
  folded into that conv: ``w' = w * m / sigma`` and
  ``b' = (b_conv - mu) * m / sigma + b``.
- The final ``fin_adjust_bn`` is a scalar BatchNorm on the cross-correlation
  output. It is kept as four scalars: ``adjust_{mult,bias,mean,sigma}``.
- Only the exemplar branch (``br1_*``) is read. Both branches share the same
  ``br_*`` parameters by construction (weight sharing), which is asserted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

CONVERTER_VERSION = "2026-10-06-siamfc-convert-v1"

# Layer chain of the exemplar branch (CFNet baseline-conv5 / SiamFC xcorr).
_CONV_BN = (
    ("br1_conv1", "br1_bn1"),
    ("br1_conv2", "br1_bn2"),
    ("br1_conv3", "br1_bn3"),
    ("br1_conv4", "br1_bn4"),
    ("br1_conv5", None),
)


@dataclass(frozen=True)
class ConvertedNet:
    weights: dict[str, np.ndarray]  # conv{i}.weight / conv{i}.bias, float32
    groups: dict[str, int]  # conv{i} -> groups
    strides: dict[str, int]  # conv{i}/pool{i} -> stride
    pool_sizes: dict[str, int]
    adjust: dict[str, float]  # mult, bias, mean, sigma
    meta: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_list(value) -> list:
    if isinstance(value, np.ndarray):
        return list(value.reshape(-1))
    return [value]


def _params_by_name(net) -> dict[str, np.ndarray]:
    return {str(p.name): np.asarray(p.value) for p in _as_list(net.params)}


def _layers_by_name(net) -> dict[str, Any]:
    return {str(layer.name): layer for layer in _as_list(net.layers)}


def _param_names(layer) -> list[str]:
    return [str(v) for v in _as_list(layer.params)]


def _block_int_pair(block, field: str) -> tuple[int, int]:
    values = [int(v) for v in np.asarray(getattr(block, field)).reshape(-1)]
    return values[0], values[1]


def convert_dagnn(net) -> ConvertedNet:
    """Convert a loaded DagNN struct (``scipy.io.loadmat(..., struct_as_record=False, squeeze_me=True)['net']``)."""
    params = _params_by_name(net)
    layers = _layers_by_name(net)

    weights: dict[str, np.ndarray] = {}
    groups: dict[str, int] = {}
    strides: dict[str, int] = {}
    pool_sizes: dict[str, int] = {}

    in_channels = 3
    for idx, (conv_name, bn_name) in enumerate(_CONV_BN, start=1):
        conv = layers[conv_name]
        pad = [int(v) for v in np.asarray(conv.block.pad).reshape(-1)]
        if any(pad):
            raise ValueError(f"{conv_name}: non-zero padding {pad} not supported")
        stride_y, stride_x = _block_int_pair(conv.block, "stride")
        if stride_y != stride_x:
            raise ValueError(f"{conv_name}: anisotropic stride")
        f_name, b_name = _param_names(conv)
        w = np.asarray(params[f_name], np.float64)  # H x W x Cin_g x Cout
        b = np.asarray(params[b_name], np.float64).reshape(-1)
        cin_g = int(w.shape[2])
        if in_channels % cin_g != 0:
            raise ValueError(f"{conv_name}: {in_channels} input channels not divisible by {cin_g}")
        n_groups = in_channels // cin_g

        if bn_name is not None:
            bn = layers[bn_name]
            m_name, bb_name, x_name = _param_names(bn)
            mult = np.asarray(params[m_name], np.float64).reshape(-1)
            beta = np.asarray(params[bb_name], np.float64).reshape(-1)
            moments = np.asarray(params[x_name], np.float64).reshape(-1, 2)
            mu, sigma = moments[:, 0], moments[:, 1]
            if np.any(sigma <= 0.0):
                raise ValueError(f"{bn_name}: non-positive sigma in moments")
            gain = mult / sigma
            w = w * gain[None, None, None, :]
            b = (b - mu) * gain + beta

        key = f"conv{idx}"
        weights[f"{key}.weight"] = np.ascontiguousarray(np.transpose(w, (3, 2, 0, 1))).astype(np.float32)
        weights[f"{key}.bias"] = b.astype(np.float32)
        groups[key] = int(n_groups)
        strides[key] = int(stride_y)
        in_channels = int(w.shape[3])

        pool_name = f"br1_pool{idx}"
        if pool_name in layers:
            pool = layers[pool_name]
            if str(pool.block.method) != "max":
                raise ValueError(f"{pool_name}: only max pooling supported")
            pool_sizes[f"pool{idx}"] = _block_int_pair(pool.block, "poolSize")[0]
            strides[f"pool{idx}"] = _block_int_pair(pool.block, "stride")[0]

    # Weight sharing sanity check: br2_* must reference the same br_* params.
    for conv_name, _ in _CONV_BN:
        twin = layers.get(conv_name.replace("br1_", "br2_"))
        if twin is not None and _param_names(twin) != _param_names(layers[conv_name]):
            raise ValueError(f"{conv_name}: exemplar/instance branches do not share weights")

    adj = layers["fin_adjust_bn"]
    m_name, bb_name, x_name = _param_names(adj)
    moments = np.asarray(params[x_name], np.float64).reshape(-1)
    adjust = {
        "mult": float(np.asarray(params[m_name]).reshape(-1)[0]),
        "bias": float(np.asarray(params[bb_name]).reshape(-1)[0]),
        "mean": float(moments[0]),
        "sigma": float(moments[1]),
    }
    if adjust["sigma"] <= 0.0:
        raise ValueError("fin_adjust_bn: non-positive sigma")

    return ConvertedNet(
        weights=weights,
        groups=groups,
        strides=strides,
        pool_sizes=pool_sizes,
        adjust=adjust,
        meta={"converter_version": CONVERTER_VERSION},
    )


def load_mat_net(path: str | Path):
    import scipy.io as sio  # noqa: PLC0415 (offline tool dependency only)

    mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    return mat["net"]


def convert_file(mat_path: str | Path, out_path: str | Path | None = None) -> Path:
    import torch  # noqa: PLC0415

    mat_path = Path(mat_path)
    out_path = Path(out_path) if out_path is not None else mat_path.with_suffix(".pth")
    converted = convert_dagnn(load_mat_net(mat_path))
    meta = dict(converted.meta)
    meta.update({"source_file": mat_path.name, "source_sha256": sha256_file(mat_path)})
    payload = {
        "state_dict": {k: torch.from_numpy(v) for k, v in converted.weights.items()},
        "groups": converted.groups,
        "strides": converted.strides,
        "pool_sizes": converted.pool_sizes,
        "adjust": converted.adjust,
        "meta": meta,
    }
    torch.save(payload, out_path)
    meta["output_sha256"] = sha256_file(out_path)
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_path


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print(__doc__)
        return 2
    out = convert_file(argv[1], argv[2] if len(argv) == 3 else None)
    print(f"wrote {out}")
    print(out.with_suffix(".json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
