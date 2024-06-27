import triton
import triton.language as tl
import torch

@triton.jit
def vecdot_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    a_ptrs = a_ptr + tl.arange(0, BLOCK_SIZE)
    b_ptrs = b_ptr + tl.arange(0, BLOCK_SIZE)
    a = tl.load(a_ptrs, mask=tl.arange(0, BLOCK_SIZE) < N)
    b = tl.load(b_ptrs, mask=tl.arange(0, BLOCK_SIZE) < N)
    c = a * b
    tl.store(c_ptr + tl.arange(0, BLOCK_SIZE), c, mask=tl.arange(0, BLOCK_SIZE) < N)

def vecdot(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda
    assert b.is_cuda
    assert a.shape == b.shape
    N = a.shape[0]
    c = torch.zeros(N, device='cuda')
    BLOCK_SIZE = triton.next_power_of_2(N)
    vecdot_kernel[(1,)](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)
    return c

def main():
    a = torch.randn(1000, device='cuda')
    b = torch.randn(1000, device='cuda')
    print(vecdot(a, b))

if __name__ == '__main__':
    main()