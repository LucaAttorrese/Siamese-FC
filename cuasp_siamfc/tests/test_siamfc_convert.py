"""MatConvNet -> PyTorch conversion: layout, groups and BatchNorm folding.

A synthetic DagNN struct (same attribute shape as scipy.io.loadmat with
struct_as_record=False) is converted, then the converted network is compared
with a direct, unfolded evaluation of MatConvNet semantics.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest

from suncubes.siamfc.convert_matconvnet import convert_dagnn

torch = pytest.importorskip("torch")
F = torch.nn.functional

# (name, k, cin_per_group, cout, stride, pool_stride or None)
_SPEC = [
    ("1", 5, 3, 8, 2, 2),
    ("2", 3, 4, 8, 1, 1),  # groups = 8 / 4 = 2
    ("3", 3, 8, 6, 1, None),
    ("4", 3, 3, 6, 1, None),  # groups 2
    ("5", 3, 3, 4, 1, None),  # groups 2, no BN
]


def _synthetic_net(rng):
    params, layers = [], []

    def add_param(name, value):
        params.append(NS(name=name, value=np.asarray(value, np.float32)))

    for name, k, cin_g, cout, stride, pool_stride in _SPEC:
        add_param(f"br_conv{name}f", rng.normal(size=(k, k, cin_g, cout)))
        add_param(f"br_conv{name}b", rng.normal(size=cout))
        conv_block = NS(pad=np.zeros(4), stride=np.array([stride, stride]))
        for br in ("br1", "br2"):
            layers.append(NS(name=f"{br}_conv{name}", block=conv_block, params=np.array([f"br_conv{name}f", f"br_conv{name}b"])))
        if name != "5":
            add_param(f"br_bn{name}m", rng.uniform(0.5, 1.5, cout))
            add_param(f"br_bn{name}b", rng.normal(size=cout))
            add_param(f"br_bn{name}x", np.stack([rng.normal(size=cout), rng.uniform(0.5, 3.0, cout)], 1))
            layers.append(NS(name=f"br1_bn{name}", block=NS(), params=np.array([f"br_bn{name}m", f"br_bn{name}b", f"br_bn{name}x"])))
        if pool_stride is not None:
            layers.append(NS(name=f"br1_pool{name}", block=NS(method="max", poolSize=np.array([3, 3]), stride=np.array([pool_stride] * 2)), params=np.array([])))
    add_param("fin_adjust_bnm", 2.5)
    add_param("fin_adjust_bnb", -4.0)
    add_param("fin_adjust_bnx", [100.0, 2000.0])
    layers.append(NS(name="fin_adjust_bn", block=NS(), params=np.array(["fin_adjust_bnm", "fin_adjust_bnb", "fin_adjust_bnx"])))
    return NS(params=np.array(params, dtype=object), layers=np.array(layers, dtype=object)), {p.name: p.value for p in params}


def _matconvnet_reference(x, raw):
    """Unfolded MatConvNet semantics: conv (HWIO filters, correlation) -> BN(m,(x-mu)/sigma,b) -> ReLU -> pool."""
    cin = x.shape[1]
    for name, _, cin_g, _, stride, pool_stride in _SPEC:
        w = torch.from_numpy(np.transpose(raw[f"br_conv{name}f"], (3, 2, 0, 1)).astype(np.float64))
        b = torch.from_numpy(raw[f"br_conv{name}b"].astype(np.float64))
        x = F.conv2d(x, w, b, stride=stride, groups=cin // cin_g)
        if name != "5":
            m = torch.from_numpy(raw[f"br_bn{name}m"].astype(np.float64)).view(1, -1, 1, 1)
            beta = torch.from_numpy(raw[f"br_bn{name}b"].astype(np.float64)).view(1, -1, 1, 1)
            mom = raw[f"br_bn{name}x"].astype(np.float64)
            mu = torch.from_numpy(mom[:, 0]).view(1, -1, 1, 1)
            sigma = torch.from_numpy(mom[:, 1]).view(1, -1, 1, 1)
            x = F.relu(m * (x - mu) / sigma + beta)
            if pool_stride is not None:
                x = F.max_pool2d(x, 3, pool_stride)
        cin = x.shape[1]
    return x


def _converted_net(net):
    from suncubes.siamfc.net import SiamFCNet  # noqa: PLC0415

    conv = convert_dagnn(net)
    payload = {
        "state_dict": {k: torch.from_numpy(v) for k, v in conv.weights.items()},
        "groups": conv.groups, "strides": conv.strides, "pool_sizes": conv.pool_sizes,
        "adjust": conv.adjust, "meta": conv.meta,
    }
    return conv, SiamFCNet(payload).double()


def test_conversion_layout_groups_and_adjust():
    net, raw = _synthetic_net(np.random.default_rng(0))
    conv = convert_dagnn(net)
    assert conv.groups == {"conv1": 1, "conv2": 2, "conv3": 1, "conv4": 2, "conv5": 2}
    assert conv.weights["conv1.weight"].shape == (8, 3, 5, 5)
    assert conv.strides["conv1"] == 2 and conv.strides["pool1"] == 2 and conv.strides["pool2"] == 1
    assert conv.adjust == {"mult": 2.5, "bias": -4.0, "mean": 100.0, "sigma": 2000.0}
    # conv5 has no BN: weights are only transposed.
    np.testing.assert_allclose(conv.weights["conv5.weight"], np.transpose(raw["br_conv5f"], (3, 2, 0, 1)), rtol=1e-6)


def test_folded_network_matches_unfolded_matconvnet_semantics():
    net, raw = _synthetic_net(np.random.default_rng(1))
    _, model = _converted_net(net)
    x = torch.from_numpy(np.random.default_rng(2).uniform(0, 255, (2, 3, 61, 61)))
    ref = _matconvnet_reference(x, raw)
    out = model.features(x)
    assert out.shape == ref.shape
    # Converted weights are stored as float32, so allow float32-level relative
    # error; a semantic error (wrong fold, variance vs sigma, layout) is O(1).
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-3)


def test_sigma_is_standard_deviation_not_variance():
    """Folding must divide by sigma: dividing by sqrt(sigma) would be the 'variance' bug."""
    net, raw = _synthetic_net(np.random.default_rng(3))
    conv = convert_dagnn(net)
    m, sigma = raw["br_bn1m"].astype(np.float64), raw["br_bn1x"][:, 1].astype(np.float64)
    expected = np.transpose(raw["br_conv1f"], (3, 2, 0, 1)) * (m / sigma)[:, None, None, None]
    np.testing.assert_allclose(conv.weights["conv1.weight"], expected, rtol=1e-5)


def test_gray_input_equals_replicated_rgb():
    net, _ = _synthetic_net(np.random.default_rng(4))
    _, model = _converted_net(net)
    gray = torch.from_numpy(np.random.default_rng(5).uniform(0, 255, (1, 1, 61, 61)))
    torch.testing.assert_close(model.features(gray), model.features(gray.expand(-1, 3, -1, -1)), rtol=1e-6, atol=0.0)


def test_score_applies_adjust_bn():
    net, _ = _synthetic_net(np.random.default_rng(6))
    _, model = _converted_net(net)
    z = torch.ones(1, 2, 3, 3, dtype=torch.float64)
    x = torch.ones(4, 2, 5, 5, dtype=torch.float64)
    s = model.score(z, x)
    assert s.shape == (4, 3, 3)
    torch.testing.assert_close(s, torch.full_like(s, 2.5 * (18.0 - 100.0) / 2000.0 - 4.0))
