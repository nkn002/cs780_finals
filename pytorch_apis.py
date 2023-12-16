import torch as th
import gp_apis

class transpose3D_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, a, dim_0, dim_1, dim_2, dim1, dim2, device0):
        res = gp_apis.gp_transpose3D(a, dim_0, dim_1, dim_2, dim1, dim2, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def transpose3D(a, dim_0, dim_1, dim_2, dim1, dim2, device0):
    return transpose3D_impl.apply(a, dim_0, dim_1, dim_2, dim1, dim2, device0)

def batch_mm(input1, input2, dim_0, dim_1, dim2, device0):
    return batch_mm_impl.apply(input1, input2, dim_0, dim_1, dim2, device0)

class batch_mm_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, dim_0, dim_1, dim2, device0):
        res = gp_apis.gp_batch_mm(input1, input2, dim_0, dim_1, dim2, device0)
        ctx.save_for_backward(input1, input2) 
        return res

    @staticmethod
    def backward(ctx, dZ):
        # Gradient with respect to the first input (input1)
        grad_input1 = torch.bmm(dZ, input2.transpose(-2, -1)).to(dZ.device)

        # Gradient with respect to the second input (input2)
        grad_input2 = torch.bmm(input1.transpose(-2, -1), dZ).to(dZ.device)

        # No gradients for the other arguments
        return grad_input1, grad_input2
        
def transpose(a, dim_0, dim_1, device0):
    return transpose_impl.apply(a, dim_0, dim_1, device0)

class transpose_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, a, dim_0, dim_1, device0):
        res = gp_apis.gp_transpose(a, dim_0, dim_1, device0)
        ctx.save_for_backward(a, dim_0, dim_1) 
        return res

    @staticmethod
    def backward(ctx, dZ):
        a, dim_0, dim_1 = ctx.saved_tensors
        # The gradient of a transpose is a transpose of the gradient
        grad_a = gp_apis.gp_transpose(dZ, dim_0, dim_1, dZ.device)

        return grad_a


def transpose(a, dim_0, dim_1, device0):
    return transpose_impl.apply(a, dim_0, dim_1, device0)

class mm_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, dim_0, dim_1, device0):
        res = gp_apis.gp_mm(input1, input2, dim_0, dim_1, device0)
        ctx.save_for_backward(input1, input2) 
        return res

    @staticmethod
    def backward(ctx, dZ):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            dim_0, dim_1 = input.shape
            weightT = gp_apis.gp_transpose(weight, weight.size(1), weight.size(0), device0)
            grad_input = gp_apis.gp_mm(dZ, weightT, dim_0, dim_1, device0)
        if ctx.needs_input_grad[1]:
            dim_0, dim_1 = weight.shape
            inputT = gp_apis.gp_transpose(input, input.size(1), input.size(0), device0)
            grad_weight = gp_apis.gp_mm(inputT, dZ, dim_0, dim_1, device0)

        return grad_input, grad_weight


def mm(input1, input2, dim_0, dim_1, device0):
    return mm_impl.apply(input1, input2, dim_0, dim_1, device0)

class sum_two_tensors_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, dim_0, dim_1, device0):
        res = gp_apis.gp_sum_two_tensors(input1, input2, dim_0, dim_1, device0)
        ctx.backward_cache = [input1, input2]
        return res

    @staticmethod
    def backward(ctx, dZ):
        return dZ, dZ

def sum_two_tensors(input1, input2, dim_0, dim_1, device0):
    return sum_two_tensors_impl.apply(input1, input2, dim_0, dim_1, device0)

class gspmmv_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input1, dim_0, dim_1, reverse, norm, device0):
        res = gp_apis.gp_gspmmv(graph, input1, dim_0, dim_1, reverse, norm, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gspmmv(graph, input1, dim_0, dim_1, reverse, norm, device0):
    return gspmmv_impl.apply(graph, input1, dim_0, dim_1, reverse, norm, device0)

class gspmmve_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input1, edge_input, dim_0, dim_1, op, reverse, device0):
        res = gp_apis.gp_gspmmve(graph, input1, edge_input, dim_0, dim_1, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gspmmve(graph, input1, edge_input, dim_0, dim_1, op, reverse, device0):
    return gspmmve_impl.apply(graph, input1, edge_input, dim_0, dim_1, op, reverse, device0)

class gspmme_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, edge_input, dim_0, op, reverse, device0):
        res = gp_apis.gp_gspmme(graph, edge_input, dim_0, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gspmme(graph, edge_input, dim_0, op, reverse, device0):
    return gspmme_impl.apply(graph, edge_input, dim_0, op, reverse, device0)

class gspmme2d_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, edge_input, dim_0, dim_1, op, reverse, device0):
        res = gp_apis.gp_gspmme2d(graph, edge_input, dim_0, dim_1, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gspmme2d(graph, edge_input, dim_0, dim_1, op, reverse, device0):
    return gspmme2d_impl.apply(graph, edge_input, dim_0, dim_1, op, reverse, device0)

class gspmmve2d_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input1, edge_input, dim_0, dim_1, dim_2, op, reverse, device0):
        res = gp_apis.gp_gspmmve2d(graph, input1, edge_input, dim_0, dim_1, dim_2, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gspmmve2d(graph, input1, edge_input, dim_0, dim_1, dim_2, op, reverse, device0):
    return gspmmve2d_impl.apply(graph, input1, edge_input, dim_0, dim_1, dim_2, op, reverse, device0)

class gsddmmve_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input_left, input_right, dim_0, op, reverse, device0):
        res = gp_apis.gp_gsddmmve(graph, input_left, input_right, dim_0, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gsddmmve(graph, input_left, input_right, dim_0, op, reverse, device0):
    return gsddmmve_impl.apply(graph, input_left, input_right, dim_0, op, reverse, device0)

class gsddmmve2d_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input_left, input_right, dim_0, dim_1, op, reverse, device0):
        res = gp_apis.gp_gsddmmve2d(graph, input_left, input_right, dim_0, dim_1, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gsddmmve2d(graph, input_left, input_right, dim_0, dim_1, op, reverse, device0):
    return gsddmmve2d_impl.apply(graph, input_left, input_right, dim_0, dim_1, op, reverse, device0)

class gsddmmvv_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input_left, input_right, dim_0, op, reverse, device0):
        res = gp_apis.gp_gsddmmvv(graph, input_left, input_right, dim_0, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gsddmmvv(graph, input_left, input_right, dim_0, op, reverse, device0):
    return gsddmmvv_impl.apply(graph, input_left, input_right, dim_0, op, reverse, device0)

class gsddmmvv2d_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input_left, input_right, dim_0, dim_1, op, reverse, device0):
        res = gp_apis.gp_gsddmmvv2d(graph, input_left, input_right, dim_0, dim_1, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gsddmmvv2d(graph, input_left, input_right, dim_0, dim_1, op, reverse, device0):
    return gsddmmvv2d_impl.apply(graph, input_left, input_right, dim_0, dim_1, op, reverse, device0)

class test_2out_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0):
        res1, res2 = gp_apis.gp_test_2out(graph, input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res1, res2

    @staticmethod
    def backward(ctx, dZ1, dZ2):
        pass #must be implemented

def test_2out(graph, input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0):
    return test_2out_impl.apply(graph, input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0)

class test3_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0):
        res1, res2 = gp_apis.gp_test3(input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res1, res2

    @staticmethod
    def backward(ctx, dZ1, dZ2):
        pass #must be implemented

def test3(input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0):
    return test3_impl.apply(input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0)

class test4_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, dim1_0, dim1_1, dim1_2, dim1_3, t, device0):
        res = gp_apis.gp_test4(input1, input2, dim1_0, dim1_1, dim1_2, dim1_3, t, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def test4(input1, input2, dim1_0, dim1_1, dim1_2, dim1_3, t, device0):
    return test4_impl.apply(input1, input2, dim1_0, dim1_1, dim1_2, dim1_3, t, device0)



