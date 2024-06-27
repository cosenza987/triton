import triton
import triton.language as tl
import torch

@triton.jit
def puzzle7_kernel(a_ptr, c_ptr, M, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = a_ptr + row_idx * M
    col_offsets = tl.arange(0, BLOCK_SIZE)
    a_ptrs = row_start_ptr + col_offsets
    a = tl.load(a_ptrs, mask=col_offsets < M)
    sum = tl.sum(a, axis=0)
    tl.store(c_ptr + row_idx, sum)

def puzzle7(a: torch.Tensor):
    assert a.is_cuda
    print(a.shape)
    N, M = a.shape
    c = torch.zeros(N, device='cuda')
    BLOCK_SIZE = triton.next_power_of_2(M)
    puzzle7_kernel[(N,)](a, c, M, BLOCK_SIZE=BLOCK_SIZE)
    return c

def main():
    a = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    a = a.to(device=torch.device('cuda'))
    print(puzzle7(a))

if __name__ == '__main__':
    main()