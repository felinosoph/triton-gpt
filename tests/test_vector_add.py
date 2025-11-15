import torch

from kernels.vector_add import vector_add

def test_vector_add_basic():
    device = "cuda"
    x = torch.randn(10_000, device=device, dtype=torch.float32)
    y = torch.randn(10_000, device=device, dtype=torch.float32)

    out = vector_add(x, y)
    ref = x + y

    max_err = (out - ref).abs().max().item()
    print("max_err basic:", max_err)
    assert max_err < 1e-5


def test_vector_add_multidim():
    device = "cuda"
    x = torch.randn(16, 32, 8, device=device, dtype=torch.float32)
    y = torch.randn(16, 32, 8, device=device, dtype=torch.float32)

    out = vector_add(x, y)
    ref = x + y

    assert out.shape == x.shape
    max_err = (out - ref).abs().max().item()
    print("max_err multidim:", max_err)
    assert max_err < 1e-5

