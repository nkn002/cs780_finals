import torch
from pytorch_apis import sum_two_tensors, mm, transpose
from copy import deepcopy
import random

device = torch.device('cuda')

for i in range(10):
    dim_0 = random.randint(2560, 7840)
    dim_1 = random.randint(2560, 7840)

    a = torch.rand((dim_0, dim_1)).cuda()
    b = torch.rand((dim_0, dim_1)).cuda()
    c = sum_two_tensors(a, b, dim_0, dim_1, device)
    if torch.allclose(c, a + b): print(f"Test {i}: Computation of SUM on GPU is correct")
    else: 
        print(f"Test {i}: Computation of SUM on GPU is wrong")
        print(c)
        print(a+b)

for i in range(10):
    row = random.randint(2560, 7840)
    col = random.randint(2560, 7840)
    common_dim = random.randint(2560, 7840)
    
    a = torch.rand((row, common_dim)).cuda()
    b = torch.rand((common_dim, col)).cuda()
    c = mm(a,b, row, col, device)

    if torch.allclose(c, a @ b): print(f"Test {i}: Computation of MM on GPU is correct")
    else: print("Computation of MM on GPU is wrong")

for i in range(10):
    row = random.randint(2560, 7840)
    col = random.randint(2560, 7840)
    common_dim = random.randint(2560, 7840)


    input = torch.rand((row, common_dim)).cuda()
    weight = torch.rand((col, common_dim)).cuda()
    output1 = input.mm(weight.t())
    dim0, dim1 = input.shape[0], weight.shape[0]

    weightT = transpose(weight, weight.size(1), weight.size(0), device)
    output2 = mm(input, weightT, dim0, dim1, device)

    if torch.allclose(output1, output2): print(f"Test {i}: Computation of MM with transpose on GPU is correct")
    else: print("Computation of MM with transpose on GPU is wrong")

