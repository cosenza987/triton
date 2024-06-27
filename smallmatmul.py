import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE: tl.constexpr):
    offs_m = tl.arange(0, BLOCK_SIZE)
    offs_n = tl.arange(0, BLOCK_SIZE)
    offs_k = tl.arange(0, BLOCK_SIZE)
    a_ptrs = a_ptr + (offs_m[:, None] * K + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * N + offs_n[None, :])
    c_ptrs = c_ptr + (offs_m[:, None] * N + offs_n[None, :])
    a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))
    b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N))
    c = tl.dot(a, b)
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def matmul(a: torch.Tensor, b: torch.Tensor):
    assert a.shape[1] == b.shape[0]
    assert a.is_cuda and b.is_cuda
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device='cuda')
    BLOCK_SIZE = triton.next_power_of_2(max(M, N, K))
    matmul_kernel[(1,)](a, b, c, M, N, K, BLOCK_SIZE=BLOCK_SIZE, num_warps=1)
    return c

def main():
    a = torch.randn(16, 16, device='cuda')
    b = torch.randn(16, 16, device='cuda')
    print(matmul(a, b))

if __name__ == '__main__':
    main()