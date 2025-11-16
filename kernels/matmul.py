import triton
import torch
import triton.language as tl

@triton.jit
def _matmul_kernel(A_ptr, B_ptr, output_ptr, M, N, K, NUM_TILES_K: tl.constexpr, BLOCK_SIZE: tl.constexpr): 
    m = tl.program_id(axis=0)
    n = tl.program_id(axis=1) 

    # let's assume we create a 
    # an access at A[am:am+BLOCK_SIZE, an:an+BLOCK_SIZE]

    am = m * BLOCK_SIZE
    bn = n * BLOCK_SIZE

    offs_m = tl.arange(0, BLOCK_SIZE)
    offs_n = tl.arange(0, BLOCK_SIZE)
    offs_k = tl.arange(0, BLOCK_SIZE)


    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)


    for k_idx in range(NUM_TILES_K): 
        k = k_idx * BLOCK_SIZE
        a_rows = (am + offs_m[:, None])  
        a_cols = k + offs_k[None, :] 
        a_idx = a_rows * K + a_cols 
        a_mask = (a_rows < M) & (a_cols < K)

        b_rows = (k + offs_k[:, None]) 
        b_cols = bn + offs_n[None, :] 
        b_idx = b_rows * N + b_cols 
        b_mask = (b_rows < K) & (b_cols < N)

        a = tl.load(A_ptr + a_idx, mask=a_mask, other=0.0)
        b = tl.load(B_ptr + b_idx, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=False)

    c_rows = (am + offs_m)[:, None]
    c_cols = bn + offs_n[None, :]
    c_idx = c_rows * N + c_cols 
    c_mask = (c_rows < M) & (c_cols < N)
    tl.store(output_ptr + c_idx, acc, mask=c_mask) 


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
    M, K = A.shape
    K2, N = B.shape

    assert A.dtype == torch.float32
    assert B.dtype == torch.float32
    assert K == K2
    assert A.is_cuda and B.is_cuda
    assert A.is_contiguous and B.is_contiguous
    assert A.device == B.device


    BLOCK_SIZE = 32 
    output = torch.empty((M, N), dtype=A.dtype, device=A.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE"]),triton.cdiv(N, meta["BLOCK_SIZE"])) 

    num_tiles_k = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
    _matmul_kernel[grid](A, B, output, M, N, K, NUM_TILES_K=num_tiles_k, BLOCK_SIZE=BLOCK_SIZE) 
    return output 


