import tensorflow as tf
import gp_apis

def transpose(a, dim;_0, dim;_1, device0):
    @tf.custom_gradient
    def _lambda():
        return transpose_real(a, dim;_0, dim;_1, device0)
    return _lambda()

def transpose_real(a, dim;_0, dim;_1, device0):
    out = gp_apis.gp_transpose(a, dim;_0, dim;_1, device0)
    def grad():
        return gp_apis.gp_transpose(a, dim;_0, dim;_1, device0)
    return out, grad

def mm(input1, input2, dim;_0, dim;_1, device0):
    @tf.custom_gradient
    def _lambda(X1, X2):
        return mm_real(X1, X2, dim;_0, dim;_1, device0)
    return _lambda(input1, input2)

def mm_real(input1, input2, dim;_0, dim;_1, device0):
    out = gp_apis.gp_mm(input1, input2, dim;_0, dim;_1, device0)
    def grad(dZ1, dZ2):
        return gp_apis.gp_mm(dZ1, dZ2, dim;_0, dim;_1, device0)
    return out, grad

def sum_two_tensors(input1, input2, dim;_0, dim;_1, device0):
    @tf.custom_gradient
    def _lambda(X1, X2):
        return sum_two_tensors_real(X1, X2, dim;_0, dim;_1, device0)
    return _lambda(input1, input2)

def sum_two_tensors_real(input1, input2, dim;_0, dim;_1, device0):
    out = gp_apis.gp_sum_two_tensors(input1, input2, dim;_0, dim;_1, device0)
    def grad(dZ1, dZ2):
        return gp_apis.gp_sum_two_tensors(dZ1, dZ2, dim;_0, dim;_1, device0)
    return out, grad

def gspmmv(graph, input1, dim_0, dim_1, reverse, norm, device0):
    @tf.custom_gradient
    def _lambda(X1):
        return gspmmv_real(graph, X1, dim_0, dim_1, reverse, norm, device0)
    return _lambda(input1)

def gspmmv_real(graph, input1, dim_0, dim_1, reverse, norm, device0):
    out = gp_apis.gp_gspmmv(graph, input1, dim_0, dim_1, 1, norm, device0)
    def grad(dZ1):
        return gp_apis.gp_gspmmv(graph, dZ1, dim_0, dim_1, 0, norm, device0)
    return out, grad

def gspmmve(graph, input1, edge_input, dim_0, dim_1, op, reverse, device0):
    @tf.custom_gradient
    def _lambda(X1, edge_X):
        return gspmmve_real(graph, X1, edge_X, dim_0, dim_1, op, reverse, device0)
    return _lambda(input1, edge_input)

def gspmmve_real(graph, input1, edge_input, dim_0, dim_1, op, reverse, device0):
    out = gp_apis.gp_gspmmve(graph, input1, edge_input, dim_0, dim_1, op, 1, device0)
    def grad(dZ1, edge_dZ):
        return gp_apis.gp_gspmmve(graph, dZ1, edge_dZ, dim_0, dim_1, op, 0, device0)
    return out, grad

def gspmme(graph, edge_input, dim_0, op, reverse, device0):
    @tf.custom_gradient
    def _lambda(edge_X):
        return gspmme_real(graph, edge_X, dim_0, op, reverse, device0)
    return _lambda(edge_input)

def gspmme_real(graph, edge_input, dim_0, op, reverse, device0):
    out = gp_apis.gp_gspmme(graph, edge_input, dim_0, op, 1, device0)
    def grad(edge_dZ):
        return gp_apis.gp_gspmme(graph, edge_dZ, dim_0, op, 0, device0)
    return out, grad

def gspmme2d(graph, edge_input, dim_0, dim_1, op, reverse, device0):
    @tf.custom_gradient
    def _lambda(edge_X):
        return gspmme2d_real(graph, edge_X, dim_0, dim_1, op, reverse, device0)
    return _lambda(edge_input)

def gspmme2d_real(graph, edge_input, dim_0, dim_1, op, reverse, device0):
    out = gp_apis.gp_gspmme2d(graph, edge_input, dim_0, dim_1, op, 1, device0)
    def grad(edge_dZ):
        return gp_apis.gp_gspmme2d(graph, edge_dZ, dim_0, dim_1, op, 0, device0)
    return out, grad

def gspmmve2d(graph, input1, edge_input, dim_0, dim_1, dim_2, op, reverse, device0):
    @tf.custom_gradient
    def _lambda(X1, edge_X):
        return gspmmve2d_real(graph, X1, edge_X, dim_0, dim_1, dim_2, op, reverse, device0)
    return _lambda(input1, edge_input)

def gspmmve2d_real(graph, input1, edge_input, dim_0, dim_1, dim_2, op, reverse, device0):
    out = gp_apis.gp_gspmmve2d(graph, input1, edge_input, dim_0, dim_1, dim_2, op, 1, device0)
    def grad(dZ1, edge_dZ):
        return gp_apis.gp_gspmmve2d(graph, dZ1, edge_dZ, dim_0, dim_1, dim_2, op, 0, device0)
    return out, grad

def gsddmmve(graph, input_left, input_right, dim_0, op, reverse, device0):
    @tf.custom_gradient
    def _lambda(X_left, X_right):
        return gsddmmve_real(graph, X_left, X_right, dim_0, op, reverse, device0)
    return _lambda(input_left, input_right)

def gsddmmve_real(graph, input_left, input_right, dim_0, op, reverse, device0):
    out = gp_apis.gp_gsddmmve(graph, input_left, input_right, dim_0, op, 1, device0)
    def grad(dZ_left, dZ_right):
        return gp_apis.gp_gsddmmve(graph, dZ_left, dZ_right, dim_0, op, 0, device0)
    return out, grad

def gsddmmve2d(graph, input_left, input_right, dim_0, dim_1, op, reverse, device0):
    @tf.custom_gradient
    def _lambda(X_left, X_right):
        return gsddmmve2d_real(graph, X_left, X_right, dim_0, dim_1, op, reverse, device0)
    return _lambda(input_left, input_right)

def gsddmmve2d_real(graph, input_left, input_right, dim_0, dim_1, op, reverse, device0):
    out = gp_apis.gp_gsddmmve2d(graph, input_left, input_right, dim_0, dim_1, op, 1, device0)
    def grad(dZ_left, dZ_right):
        return gp_apis.gp_gsddmmve2d(graph, dZ_left, dZ_right, dim_0, dim_1, op, 0, device0)
    return out, grad

def gsddmmvv(graph, input_left, input_right, dim_0, op, reverse, device0):
    @tf.custom_gradient
    def _lambda(X_left, X_right):
        return gsddmmvv_real(graph, X_left, X_right, dim_0, op, reverse, device0)
    return _lambda(input_left, input_right)

def gsddmmvv_real(graph, input_left, input_right, dim_0, op, reverse, device0):
    out = gp_apis.gp_gsddmmvv(graph, input_left, input_right, dim_0, op, 1, device0)
    def grad(dZ_left, dZ_right):
        return gp_apis.gp_gsddmmvv(graph, dZ_left, dZ_right, dim_0, op, 0, device0)
    return out, grad

def gsddmmvv2d(graph, input_left, input_right, dim_0, dim_1, op, reverse, device0):
    @tf.custom_gradient
    def _lambda(X_left, X_right):
        return gsddmmvv2d_real(graph, X_left, X_right, dim_0, dim_1, op, reverse, device0)
    return _lambda(input_left, input_right)

def gsddmmvv2d_real(graph, input_left, input_right, dim_0, dim_1, op, reverse, device0):
    out = gp_apis.gp_gsddmmvv2d(graph, input_left, input_right, dim_0, dim_1, op, 1, device0)
    def grad(dZ_left, dZ_right):
        return gp_apis.gp_gsddmmvv2d(graph, dZ_left, dZ_right, dim_0, dim_1, op, 0, device0)
    return out, grad

def test_2out(graph, input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0):
    @tf.custom_gradient
    def _lambda(X1, X2):
        return test_2out_real(graph, X1, X2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0)
    return _lambda(input1, input2)

def test_2out_real(graph, input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0):
    out = gp_apis.gp_test_2out(graph, input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, 1, device0)
    def grad(dZ1, dZ2):
        return gp_apis.gp_test_2out(graph, dZ1, dZ2, dim1_0, dim1_1, dim2_0, dim2_1, op, 0, device0)
    return out, grad

def test3(input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0):
    @tf.custom_gradient
    def _lambda(X1, X2):
        return test3_real(X1, X2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0)
    return _lambda(input1, input2)

def test3_real(input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0):
    out = gp_apis.gp_test3(input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, 1, device0)
    def grad(dZ1, dZ2):
        return gp_apis.gp_test3(dZ1, dZ2, dim1_0, dim1_1, dim2_0, dim2_1, op, 0, device0)
    return out, grad

def test4(input1, input2, device0):
    @tf.custom_gradient
    def _lambda(X1, X2):
        return test4_real(X1, X2, device0)
    return _lambda(input1, input2)

def test4_real(input1, input2, device0):
    out = gp_apis.gp_test4(input1, input2, device0)
    def grad(dZ1, dZ2):
        return gp_apis.gp_test4(dZ1, dZ2, device0)
    return out, grad

