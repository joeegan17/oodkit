"""Tests for ``oodkit.utils.array.to_numpy``."""

import numpy as np
import pytest

from oodkit.utils.array import to_numpy


def test_to_numpy_passes_ndarray_through():
    x = np.array([1.0, 2.0])
    out = to_numpy(x)
    assert out is x


def test_to_numpy_from_list():
    out = to_numpy([1, 2, 3])
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, np.array([1, 2, 3]))


def test_to_numpy_none_raises():
    with pytest.raises(ValueError, match="None"):
        to_numpy(None)


def test_to_numpy_torch_tensor():
    torch = pytest.importorskip("torch")
    t = torch.tensor([[1.0, 2.0]], requires_grad=False)
    out = to_numpy(t)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, [[1.0, 2.0]])
