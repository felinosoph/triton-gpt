import torch

from kernels.matmul import matmul


def test_matmul_basic():
    device = "cuda"
    M, K, N = 64, 96, 32

    A = torch.randn(M, K, device=device, dtype=torch.float32)
    B = torch.randn(K, N, device=device, dtype=torch.float32)

    C = matmul(A, B)
    C_ref = A @ B

    assert C.shape == (M, N)
    max_err = (C - C_ref).abs().max().item()
    print("max_err basic:", max_err)
    assert max_err < 1e-3


def test_matmul_rectangular():
    device = "cuda"
    M, K, N = 37, 53, 29  # deliberately awkward sizes

    A = torch.randn(M, K, device=device, dtype=torch.float32)
    B = torch.randn(K, N, device=device, dtype=torch.float32)

    C = matmul(A, B)
    C_ref = A @ B

    assert C.shape == (M, N)
    max_err = (C - C_ref).abs().max().item()
    print("max_err rectangular:", max_err)
    assert max_err < 1e-3

