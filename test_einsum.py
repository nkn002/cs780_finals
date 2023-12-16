from pytorch_apis import batch_mm, transpose3D
import torch
import torch.utils.benchmark as benchmark
import random

device = torch.device('cuda')
num_threads = torch.get_num_threads()
print(f'Benchmarking on {num_threads} threads')


def torch_einsum(chunk_q, chunk_k):
    result_einsum = torch.einsum('bcxd,bcyd->bcxy', (chunk_q, chunk_k))
    return result_einsum

def loop_bmm_einsum(chunk_q, chunk_k):
    chunk_q_reshaped = chunk_q.reshape(-1, chunk_q.shape[-2], chunk_q.shape[-1])
    chunk_k_reshaped = chunk_k.reshape(-1, chunk_k.shape[-2], chunk_k.shape[-1])
    # Assuming chunk_q and chunk_k are of appropriate and matching sizes for batched matrix multiplication
    batch_size = chunk_q_reshaped.size(0)
    # Determine the output size from the matrix multiplication
    output_size_x = chunk_q_reshaped.size(1)
    output_size_y = chunk_k_reshaped.size(1)

    # Pre-allocate the tensor
    res = torch.empty((batch_size, output_size_x, output_size_y), dtype=chunk_q.dtype, device=chunk_q.device)

    # Perform the batched matrix multiplication in a loop
    for i in range(batch_size):
        res[i] = torch.matmul(chunk_q_reshaped[i], chunk_k_reshaped[i].transpose(-2, -1))
    result_alternative_reshaped = res.reshape(chunk_q.shape[0], chunk_q.shape[1], chunk_q.shape[2], chunk_k.shape[2])
    return result_alternative_reshaped

def torch_bmm_einsum(chunk_q, chunk_k):
    chunk_q_reshaped = chunk_q.reshape(-1, chunk_q.shape[-2], chunk_q.shape[-1])
    chunk_k_reshaped = chunk_k.reshape(-1, chunk_k.shape[-2], chunk_k.shape[-1])

    # Perform batch matrix multiplication
    result = torch.bmm(chunk_q_reshaped, chunk_k_reshaped.transpose(-2, -1))

    # Reshape the result back to the original dimensions
    result_reshaped = result.reshape(chunk_q.shape[0], 
                                     chunk_q.shape[1], 
                                     chunk_q.shape[2], 
                                     chunk_k.shape[2])
    return result_reshaped

def cuda_bmm_einsum(chunk_q, chunk_k):
    chunk_q_reshaped = chunk_q.view(-1, chunk_q.shape[-2], chunk_q.shape[-1])
    chunk_k_reshaped = chunk_k.view(-1, chunk_k.shape[-2], chunk_k.shape[-1])
    
    batch_size = chunk_q_reshaped.size(0)
    output_size_x = chunk_q_reshaped.size(1)
    output_size_y = chunk_k_reshaped.size(1)

    result = batch_mm(chunk_q_reshaped, 
                      chunk_k_reshaped.transpose(-2, -1).contiguous(), 
                      batch_size, 
                      output_size_x, 
                      output_size_y, 
                      device)
    
    result_reshaped = result.view(chunk_q.shape[0], 
                                     chunk_q.shape[1], 
                                     chunk_q.shape[2], 
                                     chunk_k.shape[2])
    return result_reshaped



# t0 = benchmark.Timer(
#     stmt='loop_bmm_einsum(chunk_q, chunk_k)',
#     setup='from __main__ import loop_bmm_einsum',
#     num_threads=num_threads,
#     globals={'chunk_q': chunk_q, 'chunk_k': chunk_k})
label = 'Batch Matrix Multiplication'
# sub_label = 
results = []
for i in range(1,4):
    a,b,c,d,e = [i * d for d in [16,25,50,25,25]]
    print(a,b,c,d,e)
    chunk_q = torch.rand((a,b,c,e)).cuda()
    chunk_k = torch.rand((a,b,d,e)).cuda()
    sub_label = f'Q={a}x{b}x{c}x{e}, K={a}x{b}x{d}x{e}'
    t0 = benchmark.Timer(
        stmt='loop_bmm_einsum(chunk_q, chunk_k)',
        setup='from __main__ import loop_bmm_einsum',
        num_threads=num_threads,
        label=label,
        globals={'chunk_q': chunk_q, 'chunk_k': chunk_k})

    t1 = benchmark.Timer(
        stmt='cuda_bmm_einsum(chunk_q, chunk_k)',
        setup='from __main__ import cuda_bmm_einsum',
        num_threads=num_threads,
        globals={'chunk_q': chunk_q, 'chunk_k': chunk_k})


    t2 = benchmark.Timer(
        stmt='torch_einsum(chunk_q, chunk_k)',
        setup='from __main__ import torch_einsum',
        num_threads=num_threads,
        globals={'chunk_q': chunk_q, 'chunk_k': chunk_k})

    results.append(t0.timeit(100))
    results.append(t1.timeit(100))
    results.append(t2.timeit(100))
    
#     del chunk_q
#     del chunk_k
#     torch.cuda.empty_cache()
# print(results)
# compare = benchmark.Compare(results)
# print(compare)
# compare.print()

for i in range(100):
    a,b,c,d,e = random.randint(10,50), random.randint(10,50), random.randint(10,50), random.randint(10,50), random.randint(10,50)
    chunk_q = torch.rand((a,b,c,e)).cuda()
    chunk_k = torch.rand((a,b,d,e)).cuda()

    torch_res = torch_einsum(chunk_q, chunk_k)
#     loop_res = loop_bmm_einsum(chunk_q, chunk_k)
#     bmm_res = torch_bmm_einsum(chunk_q, chunk_k)
    cuda_res = cuda_bmm_einsum(chunk_q, chunk_k)

    print(torch.allclose(torch_res, cuda_res))

