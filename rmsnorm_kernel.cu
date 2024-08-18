// rmsnorm_kernel.cu

#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void rmsnorm_kernel(float* input, float* output, int size, float epsilon) {
    extern __shared__ float shared_mem[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = threadIdx.x;

    shared_mem[lane] = (tid < size) ? input[tid] * input[tid] : 0.0f;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (lane < offset) {
            shared_mem[lane] += shared_mem[lane + offset];
        }
        __syncthreads();
    }

    float rms = sqrtf(shared_mem[0] / size + epsilon);

    if (tid < size) {
        output[tid] = input[tid] / rms;
    }
}

void rmsnorm_cuda(torch::Tensor input, torch::Tensor output, float epsilon) {
    int size = input.size(0);
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    rmsnorm_kernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size, epsilon);

    cudaDeviceSynchronize();
}
