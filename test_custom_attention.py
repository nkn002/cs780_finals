from sliding_chunks import sliding_chunks_matmul_qk
import torch
import torch.utils.benchmark as benchmark
import random

device = torch.device('cuda')
num_threads = torch.get_num_threads()
print(f'Benchmarking on {num_threads} threads')

a,b,c,d = 16, 256, 64, 12
print(a,b,c,d)
w, padding_value = 128, 0
chunk_q = torch.rand((a,b,c,d), requires_grad=True).cuda()
chunk_k = torch.rand((a,b,c,d), requires_grad=True).cuda()


t0 = benchmark.Timer(
    stmt='sliding_chunks_matmul_qk(chunk_q, chunk_k, w, padding_value, custom)',
    setup='from sliding_chunks import sliding_chunks_matmul_qk',
    num_threads=num_threads,
    globals={'chunk_q': chunk_q, 
             'chunk_k': chunk_k,
             'w': w,
             'padding_value': padding_value,
             'custom': False
            })

t1 = benchmark.Timer(
    stmt='sliding_chunks_matmul_qk(chunk_q, chunk_k, w, padding_value, custom)',
    setup='from sliding_chunks import sliding_chunks_matmul_qk',
    num_threads=num_threads,
    globals={'chunk_q': chunk_q, 'chunk_k': chunk_k,
            'w': w,
            'padding_value': padding_value,
            'custom': True})

print("Using custom BMM: False", t0.timeit(100))
print("Using custom BMM: True", t1.timeit(100))

a,b,c,d = 32, 256, 64, 12
print(a,b,c,d)
w, padding_value = 128, 0
chunk_q = torch.rand((a,b,c,d), requires_grad=True).cuda()
chunk_k = torch.rand((a,b,c,d), requires_grad=True).cuda()


t0 = benchmark.Timer(
    stmt='sliding_chunks_matmul_qk(chunk_q, chunk_k, w, padding_value, custom)',
    setup='from sliding_chunks import sliding_chunks_matmul_qk',
    num_threads=num_threads,
    globals={'chunk_q': chunk_q, 
             'chunk_k': chunk_k,
             'w': w,
             'padding_value': padding_value,
             'custom': False
            })

t1 = benchmark.Timer(
    stmt='sliding_chunks_matmul_qk(chunk_q, chunk_k, w, padding_value, custom)',
    setup='from sliding_chunks import sliding_chunks_matmul_qk',
    num_threads=num_threads,
    globals={'chunk_q': chunk_q, 'chunk_k': chunk_k,
            'w': w,
            'padding_value': padding_value,
            'custom': True})

print("Using custom BMM: False", t0.timeit(100))
print("Using custom BMM: True", t1.timeit(100))

for i in range(100):
    a,b,c,d = random.randint(8,64), 256, 64, 12
    chunk_q = torch.rand((a,b,c,d), requires_grad=True).cuda()
    chunk_k = torch.rand((a,b,c,d), requires_grad=True).cuda()

    torch_res = sliding_chunks_matmul_qk(chunk_q, chunk_k, 128, 0, custom=False)
#     loop_res = loop_bmm_einsum(chunk_q, chunk_k)
#     bmm_res = torch_bmm_einsum(chunk_q, chunk_k)
    cuda_res = sliding_chunks_matmul_qk(chunk_q, chunk_k, 128, 0, custom=True)

    print('Forward pass of custom and OG are equal:', torch.allclose(torch_res, cuda_res))
 
# test_q1 = torch.ones((a,b,c,d), requires_grad=True).cuda()#chunk_q.clone()
# test_k1 = torch.ones((a,b,c,d), requires_grad=True).cuda()#chunk_k.clone()
# print(test_q1.grad, test_k1.grad)

# for i in range(100):
#     # Forward pass through the function
#     output = sliding_chunks_matmul_qk(test_q1, test_k1, 128, 0, custom=False)

#     # Assuming the output is not a scalar, create a scalar to backpropagate from
#     # For example, if output is a tensor, you might sum up its elements
#     loss = output.sum()

#     # Backward pass (compute gradients)
#     loss.backward()

#     # If you need to update the values of test_q1 and test_k1 based on the gradients,
#     # you should do this here, typically with an optimizer.
#     # However, if you only need to accumulate gradients, the above is sufficient.

#     # Zero the gradients after updating
#     test_q1.grad.zero_()
#     test_k1.grad.zero_()
    
# test_q2 = torch.ones((a,b,c,d), requires_grad=True).cuda()#chunk_q.clone()
# test_k2 = torch.ones((a,b,c,d), requires_grad=True).cuda()#chunk_k.clone()

# for i in range(100):
#     # Forward pass through the function
#     output = sliding_chunks_matmul_qk(test_q1, test_k1, 128, 0, custom=True)

#     # Assuming the output is not a scalar, create a scalar to backpropagate from
#     # For example, if output is a tensor, you might sum up its elements
#     loss = output.sum()

#     # Backward pass (compute gradients)
#     loss.backward()

#     # If you need to update the values of test_q1 and test_k1 based on the gradients,
#     # you should do this here, typically with an optimizer.
#     # However, if you only need to accumulate gradients, the above is sufficient.

#     # Zero the gradients after updating
#     test_q2.grad.zero_()
#     test_k2.grad.zero_()
    
# print('torch.allclose(test_q1, test_q2)', torch.allclose(test_q1, test_q2))
# print('torch.allclose(test_k1, test_k2)', torch.allclose(test_k1, test_k2))
