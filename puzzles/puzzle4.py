import triton
import triton.language as tl
import torch

@triton.jit
def puzzle4_kernel(output_ptr, output_row_stride, a_ptr, b_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_index = tl.program_id(0)
    col_offset = tl.arange(0, BLOCK_SIZE)
    row_ptrs = a_ptr + col_offset
    row = tl.load(row_ptrs)
    col = tl.load(b_ptr + row_index)
    row_out = row + col
    output_row_ptr = output_ptr + row_index * output_row_stride
    output_ptrs = output_row_ptr + col_offset
    tl.store(output_ptrs, row_out, mask = col_offset < n_cols)

def puzzle4(a: torch.Tensor, b: torch.Tensor):
    rows, cols = (b.size(0), a.size(0))
    output = torch.zeros(rows, cols, device='cuda')
    assert a.is_cuda and b.is_cuda and output.is_cuda
    block_size = triton.next_power_of_2(cols)
    num_warps = 4
    if block_size >= 2048:
        num_warps = 8
    if block_size >= 4096:
        num_warps = 16
    puzzle4_kernel[(rows, )](output, output.stride(0), a, b, cols, num_warps=num_warps, BLOCK_SIZE=block_size)
    return output

# this works, however triton complains if I try to do everything at once...
def main():
    a = torch.randn(5000, 1, device='cuda')
    b = torch.randn(2048, 1, device='cuda')
    print(puzzle4(a, b))

if __name__ == '__main__':
    main()