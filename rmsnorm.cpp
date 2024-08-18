// rmsnorm.cpp

#include <torch/extension.h>

// 声明CUDA函数
void rmsnorm_cuda(torch::Tensor input, torch::Tensor output, float epsilon);

// PyTorch接口
torch::Tensor rmsnorm(torch::Tensor input, float epsilon) {
    auto output = torch::zeros_like(input);
    rmsnorm_cuda(input, output, epsilon);
    return output;
}

// 定义模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm", &rmsnorm, "RMSNorm CUDA implementation");
}
