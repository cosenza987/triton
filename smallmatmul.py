import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)
    a_ptrs = a_ptr + (offs_m[:, None] * K + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * N + offs_n[None, :])
    c_ptrs = c_ptr + (offs_m[:, None] * N + offs_n[None, :])
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    c = tl.dot(a, b)
    tl.store(c_ptrs, c)

def matmul(a: torch.Tensor, b: torch.Tensor):
    assert a.shape[1] == b.shape[0]
    assert a.is_cuda and b.is_cuda
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device='cuda')
    matmul_kernel[(1,)](a, b, c, M, N, K, num_warps=1)
    return c

def main():
    a = torch.randn(16, 16, device='cuda')
    b = torch.randn(16, 16, device='cuda')
    print(matmul(a, b))

if __name__ == '__main__':
    main()