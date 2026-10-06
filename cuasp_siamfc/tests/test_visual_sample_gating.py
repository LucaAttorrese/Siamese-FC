"""Tests for cuasp.py's _is_new_visual_sample, the helper main() uses to
implement root README.md's "Non-negotiable interface semantics" (a command
update must be gated by new visual sample identity, not dispatched
unconditionally on every poll).
"""

import pytest


@pytest.fixture
def cuasp(cuasp_module):
    return cuasp_module


def test_first_sample_is_always_new(cuasp):
    assert cuasp._is_new_visual_sample(None, 123) is True


def test_same_capture_ns_is_not_new(cuasp):
    assert cuasp._is_new_visual_sample(123, 123) is False


def test_different_capture_ns_is_new(cuasp):
    assert cuasp._is_new_visual_sample(123, 124) is True
