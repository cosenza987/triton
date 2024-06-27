import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) #identifying which program we're running
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    N = output.numel()
    #specifying how many iterations to go over the elements
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, N, BLOCK_SIZE=1024)
    return output

def main():
    f = open("in.txt", "r")
    al = list(map(int, f.readline().split()))
    bl = list(map(int, f.readline().split()))
    f.close()
    a = torch.Tensor(al)
    a = a.to(device=torch.device('cuda'))
    b = torch.Tensor(bl)
    b = b.to(device=torch.device('cuda'))
    f = open("out.txt", "w")
    f.write(str(add(a, b).tolist()))

if __name__ == '__main__':
    main()