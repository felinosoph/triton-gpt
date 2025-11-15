import torch
import triton.language as tl
import triton


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    result = x + y
    tl.store(output_ptr + offsets, result, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    x, y: 1D or nD tensors on CUDA, same shape, same dtype (float32 is fine for now).
    Returns: tensor of same shape with elementwise x + y computed via your Triton kernel.
    """
    assert x.is_cuda and y.is_cuda
    assert x.device == y.device
    assert x.shape == y.shape
    assert x.dtype == y.dtype

    x_flat = x.contiguous().view(-1)
    y_flat = y.contiguous().view(-1)
    

    BLOCK_SIZE = 1024
    n = x_flat.numel()

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    output_flat = torch.empty_like(x_flat)

    _vector_add_kernel[grid](x_flat, y_flat, output_flat, n, BLOCK_SIZE=BLOCK_SIZE)

    output = output_flat.view_as(x)
    return output
