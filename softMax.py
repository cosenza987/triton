import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0) # getting where we are
    row_start_ptr = input_ptr + row_idx * input_row_stride # how much we need to increase the pointer
    col_offsets = tl.arange(0, BLOCK_SIZE) 
    input_ptrs = row_start_ptr + col_offsets 
    #loading row into sram
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    num = tl.exp(row_minus_max)
    den = tl.sum(num, axis=0)
    output = num / den
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, output, mask=col_offsets < n_cols) 

def softmax(x: torch.Tensor):
    print(x.shape)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # amount of threads per row -- seems like the threads sync when performing global operations
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    y = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda
    softmax_kernel[(n_rows, )](y, x, x.stride(0), y.stride(0), n_cols, num_warps=num_warps, BLOCK_SIZE=BLOCK_SIZE)
    return y

def main():
    f = open("in.txt", "r")
    al = []
    al.append(list(map(int, f.readline().split())))
    al.append(list(map(int, f.readline().split())))
    f.close()
    a = torch.Tensor(al)
    a = a.to(device=torch.device('cuda'))
    f = open("out2.txt", "w")
    f.write(str(softmax(a).tolist()))

if __name__ == '__main__':
    main()