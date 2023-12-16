inline void export_kernel(py::module &m) { 
    m.def("transpose3D",[](py::capsule& a, py::capsule& output, int dim1, int dim2){
        array3d_t<float> a_array = capsule_to_array3d(a);
        array3d_t<float> output_array = capsule_to_array3d(output);
    return transpose3D(a_array, output_array, dim1, dim2);
    }
  );
    m.def("batch_mm",[](py::capsule& a, py::capsule& b, py::capsule& output){
        array3d_t<float> a_array = capsule_to_array3d(a);
        array3d_t<float> b_array = capsule_to_array3d(b);
        array3d_t<float> output_array = capsule_to_array3d(output);
    return batch_mm(a_array, b_array, output_array);
    }
  );
    m.def("transpose",[](py::capsule& a, py::capsule& output){
        array2d_t<float> a_array = capsule_to_array2d(a);
        array2d_t<float> output_array = capsule_to_array2d(output);
    return transpose(a_array, output_array);
    }
  );
    m.def("mm",[](py::capsule& input1, py::capsule& input2, py::capsule& output){
        array2d_t<float> input1_array = capsule_to_array2d(input1);
        array2d_t<float> input2_array = capsule_to_array2d(input2);
        array2d_t<float> output_array = capsule_to_array2d(output);
    return mm(input1_array, input2_array, output_array);
    }
  );
    m.def("sum_two_tensors",[](py::capsule& input1, py::capsule& input2, py::capsule& output){
        array2d_t<float> input1_array = capsule_to_array2d(input1);
        array2d_t<float> input2_array = capsule_to_array2d(input2);
        array2d_t<float> output_array = capsule_to_array2d(output);
    return sum_two_tensors(input1_array, input2_array, output_array);
    }
  );
    m.def("gspmmv",[](graph_t& graph, py::capsule& input1, py::capsule& output, bool reverse, bool norm){
        array2d_t<float> input1_array = capsule_to_array2d(input1);
        array2d_t<float> output_array = capsule_to_array2d(output);
    return gspmmv(graph, input1_array, output_array, reverse, norm);
    }
  );
    m.def("gspmmve",[](graph_t& graph, py::capsule& input1, py::capsule& edge_input, py::capsule& output, op_t op, bool reverse){
        array2d_t<float> input1_array = capsule_to_array2d(input1);
        array1d_t<float> edge_input_array = capsule_to_array1d(edge_input);
        array2d_t<float> output_array = capsule_to_array2d(output);
    return gspmmve(graph, input1_array, edge_input_array, output_array, op, reverse);
    }
  );
    m.def("gspmme",[](graph_t& graph, py::capsule& edge_input, py::capsule& output, op_t op, bool reverse){
        array1d_t<float> edge_input_array = capsule_to_array1d(edge_input);
        array1d_t<float> output_array = capsule_to_array1d(output);
    return gspmme(graph, edge_input_array, output_array, op, reverse);
    }
  );
    m.def("gspmme2d",[](graph_t& graph, py::capsule& edge_input, py::capsule& output, op_t op, bool reverse){
        array2d_t<float> edge_input_array = capsule_to_array2d(edge_input);
        array2d_t<float> output_array = capsule_to_array2d(output);
    return gspmme2d(graph, edge_input_array, output_array, op, reverse);
    }
  );
    m.def("gspmmve2d",[](graph_t& graph, py::capsule& input1, py::capsule& edge_input, py::capsule& output, op_t op, bool reverse){
        array3d_t<float> input1_array = capsule_to_array3d(input1);
        array2d_t<float> edge_input_array = capsule_to_array2d(edge_input);
        array3d_t<float> output_array = capsule_to_array3d(output);
    return gspmmve2d(graph, input1_array, edge_input_array, output_array, op, reverse);
    }
  );
    m.def("gsddmmve",[](graph_t& graph, py::capsule& input_left, py::capsule& input_right, py::capsule& output, op_t op, bool reverse){
        array1d_t<float> input_left_array = capsule_to_array1d(input_left);
        array1d_t<float> input_right_array = capsule_to_array1d(input_right);
        array1d_t<float> output_array = capsule_to_array1d(output);
    return gsddmmve(graph, input_left_array, input_right_array, output_array, op, reverse);
    }
  );
    m.def("gsddmmve2d",[](graph_t& graph, py::capsule& input_left, py::capsule& input_right, py::capsule& output, op_t op, bool reverse){
        array2d_t<float> input_left_array = capsule_to_array2d(input_left);
        array2d_t<float> input_right_array = capsule_to_array2d(input_right);
        array2d_t<float> output_array = capsule_to_array2d(output);
    return gsddmmve2d(graph, input_left_array, input_right_array, output_array, op, reverse);
    }
  );
    m.def("gsddmmvv",[](graph_t& graph, py::capsule& input_left, py::capsule& input_right, py::capsule& output, op_t op, bool reverse){
        array2d_t<float> input_left_array = capsule_to_array2d(input_left);
        array2d_t<float> input_right_array = capsule_to_array2d(input_right);
        array1d_t<float> output_array = capsule_to_array1d(output);
    return gsddmmvv(graph, input_left_array, input_right_array, output_array, op, reverse);
    }
  );
    m.def("gsddmmvv2d",[](graph_t& graph, py::capsule& input_left, py::capsule& input_right, py::capsule& output, op_t op, bool reverse){
        array3d_t<float> input_left_array = capsule_to_array3d(input_left);
        array3d_t<float> input_right_array = capsule_to_array3d(input_right);
        array2d_t<float> output_array = capsule_to_array2d(output);
    return gsddmmvv2d(graph, input_left_array, input_right_array, output_array, op, reverse);
    }
  );
    m.def("test_2out",[](graph_t& graph, py::capsule& input1, py::capsule& input2, py::capsule& output1, py::capsule& output2, op_t op, bool reverse){
        array2d_t<float> input1_array = capsule_to_array2d(input1);
        array2d_t<float> input2_array = capsule_to_array2d(input2);
        array2d_t<float> output1_array = capsule_to_array2d(output1);
        array2d_t<float> output2_array = capsule_to_array2d(output2);
    return test_2out(graph, input1_array, input2_array, output1_array, output2_array, op, reverse);
    }
  );
    m.def("test3",[](py::capsule& input1, py::capsule& input2, py::capsule& output1, py::capsule& output2, op_t op, bool reverse){
        array2d_t<float> input1_array = capsule_to_array2d(input1);
        array2d_t<float> input2_array = capsule_to_array2d(input2);
        array2d_t<float> output1_array = capsule_to_array2d(output1);
        array2d_t<float> output2_array = capsule_to_array2d(output2);
    return test3(input1_array, input2_array, output1_array, output2_array, op, reverse);
    }
  );
    m.def("test4",[](py::capsule& input1, py::capsule& input2, py::capsule& output1, int t){
        array3d_t<float> input1_array = capsule_to_array3d(input1);
        array4d_t<float> input2_array = capsule_to_array4d(input2);
        array4d_t<float> output1_array = capsule_to_array4d(output1);
    return test4(input1_array, input2_array, output1_array, t);
    }
  );
}