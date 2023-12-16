import torch as th
import torch.utils.dlpack
import graphpy as gpk
def gp_transpose(a, dim1_0, dim1_1, device0):
    a_dl = th.utils.dlpack.to_dlpack(a)
    res1 = th.zeros(dim1_0, dim1_1, device = device0)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    gpk.transpose(a_dl, res_dl1)
    return res1
def gp_mm(input1, input2, dim1_0, dim1_1, device0):
    input1_dl = th.utils.dlpack.to_dlpack(input1)
    input2_dl = th.utils.dlpack.to_dlpack(input2)
    res1 = th.zeros(dim1_0, dim1_1, device = device0)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    gpk.mm(input1_dl, input2_dl, res_dl1)
    return res1
def gp_sum_two_tensors(input1, input2, dim1_0, dim1_1, device0):
    input1_dl = th.utils.dlpack.to_dlpack(input1)
    input2_dl = th.utils.dlpack.to_dlpack(input2)
    res1 = th.zeros(dim1_0, dim1_1, device = device0)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    gpk.sum_two_tensors(input1_dl, input2_dl, res_dl1)
    return res1
def gp_gspmmv(graph, input1, dim1_0, dim1_1, reverse, norm, device0):
    input1_dl = th.utils.dlpack.to_dlpack(input1)
    res1 = th.zeros(dim1_0, dim1_1, device = device0)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    gpk.gspmmv(graph, input1_dl, res_dl1, reverse, norm)
    return res1
def gp_gspmmve(graph, input1, edge_input, dim1_0, dim1_1, op, reverse, device0):
    input1_dl = th.utils.dlpack.to_dlpack(input1)
    edge_input_dl = th.utils.dlpack.to_dlpack(edge_input)
    res1 = th.zeros(dim1_0, dim1_1, device = device0)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    gpk.gspmmve(graph, input1_dl, edge_input_dl, res_dl1, op, reverse)
    return res1
def gp_gspmme(graph, edge_input, dim1_0, op, reverse, device0):
    edge_input_dl = th.utils.dlpack.to_dlpack(edge_input)
    res1 = th.zeros(dim1_0, device = device0)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    gpk.gspmme(graph, edge_input_dl, res_dl1, op, reverse)
    return res1
def gp_gspmme2d(graph, edge_input, dim1_0, dim1_1, op, reverse, device0):
    edge_input_dl = th.utils.dlpack.to_dlpack(edge_input)
    res1 = th.zeros(dim1_0, dim1_1, device = device0)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    gpk.gspmme2d(graph, edge_input_dl, res_dl1, op, reverse)
    return res1
def gp_gspmmve2d(graph, input1, edge_input, dim1_0, dim1_1, dim1_2, op, reverse, device0):
    input1_dl = th.utils.dlpack.to_dlpack(input1)
    edge_input_dl = th.utils.dlpack.to_dlpack(edge_input)
    res1 = th.zeros(dim1_0, dim1_1, dim1_2, device = device0)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    gpk.gspmmve2d(graph, input1_dl, edge_input_dl, res_dl1, op, reverse)
    return res1
def gp_gsddmmve(graph, input_left, input_right, dim1_0, op, reverse, device0):
    input_left_dl = th.utils.dlpack.to_dlpack(input_left)
    input_right_dl = th.utils.dlpack.to_dlpack(input_right)
    res1 = th.zeros(dim1_0, device = device0)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    gpk.gsddmmve(graph, input_left_dl, input_right_dl, res_dl1, op, reverse)
    return res1
def gp_gsddmmve2d(graph, input_left, input_right, dim1_0, dim1_1, op, reverse, device0):
    input_left_dl = th.utils.dlpack.to_dlpack(input_left)
    input_right_dl = th.utils.dlpack.to_dlpack(input_right)
    res1 = th.zeros(dim1_0, dim1_1, device = device0)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    gpk.gsddmmve2d(graph, input_left_dl, input_right_dl, res_dl1, op, reverse)
    return res1
def gp_gsddmmvv(graph, input_left, input_right, dim1_0, op, reverse, device0):
    input_left_dl = th.utils.dlpack.to_dlpack(input_left)
    input_right_dl = th.utils.dlpack.to_dlpack(input_right)
    res1 = th.zeros(dim1_0, device = device0)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    gpk.gsddmmvv(graph, input_left_dl, input_right_dl, res_dl1, op, reverse)
    return res1
def gp_gsddmmvv2d(graph, input_left, input_right, dim1_0, dim1_1, op, reverse, device0):
    input_left_dl = th.utils.dlpack.to_dlpack(input_left)
    input_right_dl = th.utils.dlpack.to_dlpack(input_right)
    res1 = th.zeros(dim1_0, dim1_1, device = device0)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    gpk.gsddmmvv2d(graph, input_left_dl, input_right_dl, res_dl1, op, reverse)
    return res1
def gp_test_2out(graph, input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0):
    input1_dl = th.utils.dlpack.to_dlpack(input1)
    input2_dl = th.utils.dlpack.to_dlpack(input2)
    res1 = th.zeros(dim1_0, dim1_1, device = device0)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    res2 = th.zeros(dim2_0, dim2_1, device = device0)
    res_dl2 = th.utils.dlpack.to_dlpack(res2)
    gpk.test_2out(graph, input1_dl, input2_dl, res_dl1, res_dl2, op, reverse)
    return res1, res2
def gp_test3(input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0):
    input1_dl = th.utils.dlpack.to_dlpack(input1)
    input2_dl = th.utils.dlpack.to_dlpack(input2)
    res1 = th.zeros(dim1_0, dim1_1, device = device0)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    res2 = th.zeros(dim2_0, dim2_1, device = device0)
    res_dl2 = th.utils.dlpack.to_dlpack(res2)
    gpk.test3(input1_dl, input2_dl, res_dl1, res_dl2, op, reverse)
    return res1, res2
def gp_test4(input1, input2, dim1_0, dim1_1, dim1_2, dim1_3, t, device0):
    input1_dl = th.utils.dlpack.to_dlpack(input1)
    input2_dl = th.utils.dlpack.to_dlpack(input2)
    res1 = th.zeros(dim1_0, dim1_1, dim1_2, dim1_3, device = device0)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    gpk.test4(input1_dl, input2_dl, res_dl1, t)
    return res1
