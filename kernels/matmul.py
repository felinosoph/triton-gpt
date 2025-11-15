import triton
import torch
import triton.language as tl
from triton.language.core import dtype 

@triton.jit
def _matmul_kernel(A_ptr, B_ptr, output_ptr, M, N, K, TILE_DIM: tl.constexpr): 
    pass 

def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute A @ B using a Triton kernel.

    Constraints (for now):
      - A: (M, K), B: (K, N)
      - dtype: float32
      - device: CUDA
      - contiguous
    Returns:
      - C: (M, N) = A @ B
    """
    M, K = A.shape() 
    K2, N = B.shape()

    assert A.dtype == B.dtype 
    assert K == K2
    assert A.is_cuda and B.is_cuda
    assert A.is_contiguous and B.is_contiguous
    assert A.device == B.device

    output = torch.empty((M, N), dtype=torch.float, device=A.device)
    


