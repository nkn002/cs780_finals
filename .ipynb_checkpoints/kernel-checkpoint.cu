#include "kernel.h"

__global__ void transpose_3d(float *input, float *output, int size0, int size1, int size2, int dim1, int dim2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Check if the thread indices are within the bounds of the tensor
    if (i < size0 && j < size1 && k < size2) {
        // Calculate the linear index for the input and output tensors
        int in_idx = i * size1 * size2 + j * size2 + k;
        int out_idx;

        // Transpose the dimensions
        if (dim1 == 0 && dim2 == 1) {         // Transposing dimensions 0 and 1
            out_idx = j * size0 * size2 + i * size2 + k;
        } else if (dim1 == 0 && dim2 == 2) {  // Transposing dimensions 0 and 2
            out_idx = k * size0 * size1 + j * size1 + i;
        } else if (dim1 == 1 && dim2 == 2) {  // Transposing dimensions 1 and 2
            out_idx = i * size2 * size1 + k * size1 + j;
        } else {
            // Handle invalid dimension indices
            out_idx = in_idx;  // No transposition
        }

        // Perform the transposition
        output[out_idx] = input[in_idx];
    }
}


__global__ void batched_matmul(float *a, float *b, float *c, int batch_size, int m, int n, int k) { 
    unsigned int batch = blockIdx.z; // Add a batch index
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y; 
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(batch < batch_size && col < k && row < m) {
        float sum = 0;
        for(int i = 0; i < n; i++) {
            sum += a[batch * m * n + row * n + i] * b[batch * n * k + i * k + col];
        }
        c[batch * m * k + row * k + col] = sum;
    }
}


/**
 * CUDA kernel for matrix transpose operation.
 * Reference: https://github.com/lzhengchun/matrix-cuda/blob/master/matrix_cuda.cu
 */
__global__ void transpose_kernel(float* mat_in, float* mat_out, unsigned int rows, unsigned int cols) 
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) 
    {
        unsigned int pos = idy * cols + idx;
        unsigned int trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}


/**
 * CUDA kernel for matrix multiplication operation.
 * Reference: https://github.com/lzhengchun/matrix-cuda/blob/master/matrix_cuda.cu
 */
__global__ void matmul(float *a,float *b, float *c, int m, int n, int k)
{ 
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y; 
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

/**
 * CUDA kernel for matrix summation operation.
 */
__global__ void sumMatrices(float* A, float* B, float* C, int numRows, int numCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows && col < numCols) {
        int index = row * numCols + col;
        C[index] = A[index] + B[index];
    }
}

void batch_mm(array3d_t<float>& a, array3d_t<float>& b, array3d_t<float>& output) {
    int batch_size = a.matrix_count;
    int m = a.row_count;
    int n = a.col_count;
    int k = b.col_count;
    
    // Adjust the grid dimensions to account for the batch size
    dim3 dimBlock(16, 16);
    dim3 dimGrid((dimBlock.x + k - 1) / dimBlock.x, (dimBlock.y + m - 1) / dimBlock.y, batch_size);

    // Call the modified kernel
    batched_matmul<<<dimGrid, dimBlock>>>(a.data_ptr, b.data_ptr, output.data_ptr, batch_size, m, n, k);
    // cudaDeviceSynchronize();
}

// Host function to call the kernel
void transpose3D(array3d_t<float>& a, array3d_t<float>& output, int dim1, int dim2) {
    // Define the number of blocks and threads per block
    int size0 = a.matrix_count;
    int size1 = a.row_count;
    int size2 = a.col_count;

dim3 threadsPerBlock(16, 16, 16);
    dim3 numBlocks((size0 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (size1 + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (size2 + threadsPerBlock.z - 1) / threadsPerBlock.z);

    // Launch the kernel
    transpose_3d<<<numBlocks, threadsPerBlock>>>(a.data_ptr, output.data_ptr, size0, size1, size2, dim1, dim2);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
}

void transpose(array2d_t<float>& a, array2d_t<float>& output){
    int row = a.row_count;
    int col = a.col_count;
    dim3 dimBlock(16, 16);
    dim3 dimGrid((dimBlock.x + col - 1) / dimBlock.x, (dimBlock.y + row - 1) / dimBlock.y);
    transpose_kernel<<<dimGrid, dimBlock>>>(a.data_ptr, output.data_ptr, row, col);
//    cudaDeviceSynchronize();
}

void mm(array2d_t<float>& a, array2d_t<float>& b, array2d_t<float>& output){
    int m = a.row_count;
    int n = a.col_count;
    int k = b.col_count;
    
    dim3 dimBlock(16, 16);
    dim3 dimGrid((dimBlock.x + k - 1) / dimBlock.x, (dimBlock.y + m - 1) / dimBlock.y);
    matmul<<<dimGrid, dimBlock>>>(a.data_ptr, b.data_ptr, output.data_ptr, m, n, k);
//    cudaDeviceSynchronize();
}

void sum_two_tensors(array2d_t<float>& a, array2d_t<float>& b, array2d_t<float>& output){
    int m = a.row_count;
    int n = a.col_count;
    dim3 dimBlock(16, 16);
    dim3 dimGrid((dimBlock.x + n - 1) / dimBlock.x, (dimBlock.y + m - 1) / dimBlock.y);
    sumMatrices<<<dimGrid, dimBlock>>>(a.data_ptr, b.data_ptr, output.data_ptr, m, n);
//    cudaDeviceSynchronize();
}
void gspmmv(graph_t& graph, array2d_t<float>& input1, array2d_t<float>& output, bool reverse, bool norm){;}
void gspmmve(graph_t& graph, array2d_t<float>& input1, array1d_t<float>& edge_input, array2d_t<float>& output, op_t op, bool reverse){;}
void gspmme(graph_t& graph, array1d_t<float>& edge_input, array1d_t<float>& output, op_t op, bool reverse){;}
void gspmme2d(graph_t& graph, array2d_t<float>& edge_input, array2d_t<float>& output, op_t op, bool reverse){;}
void gspmmve2d(graph_t& graph, array3d_t<float>& input1, array2d_t<float>& edge_input, array3d_t<float>& output, op_t op, bool reverse){;}
void gsddmmve(graph_t& graph, array1d_t<float>& input_left, array1d_t<float>& input_right, array1d_t<float>& output, op_t op, bool reverse){;}
void gsddmmve2d(graph_t& graph, array2d_t<float>& input_left, array2d_t<float>& input_right, array2d_t<float>& output, op_t op, bool reverse){;}
void gsddmmvv(graph_t& graph, array2d_t<float>& input_left, array2d_t<float>& input_right, array1d_t<float>& output, op_t op, bool reverse){;}
void gsddmmvv2d(graph_t& graph, array3d_t<float>& input_left, array3d_t<float>& input_right, array2d_t<float>& output, op_t op, bool reverse){;}
void test_2out(graph_t& graph, array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output1, array2d_t<float>& output2, op_t op, bool reverse){;}
void test3(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output1, array2d_t<float>& output2, op_t op, bool reverse){;}
void test4(array3d_t<float>& input1, array4d_t<float>& input2, array4d_t<float>& output1, int t){;}
