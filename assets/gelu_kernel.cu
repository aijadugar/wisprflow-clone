#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <vector>

#define SQRT_2_OVER_PI 0.7978845608028654f
#define GELU_COEFF 0.044715f


// ------------------------------------------------------------
// GELU forward
// ------------------------------------------------------------

__device__ __forceinline__ float gelu_fwd(float x) {
    float x3 = x * x * x;

    float inner =
        SQRT_2_OVER_PI *
        (x + GELU_COEFF * x3);

    return 0.5f * x * (1.0f + tanhf(inner));
}


// ------------------------------------------------------------
// GELU derivative
// ------------------------------------------------------------

__device__ __forceinline__ float gelu_bwd(float x) {
    float x2 = x * x;
    float x3 = x2 * x;

    float inner =
        SQRT_2_OVER_PI *
        (x + GELU_COEFF * x3);

    float tanh_inner = tanhf(inner);

    float sech2 =
        1.0f - tanh_inner * tanh_inner;

    float d_inner =
        SQRT_2_OVER_PI *
        (1.0f + 3.0f * GELU_COEFF * x2);

    return 0.5f *
           (1.0f +
            tanh_inner +
            x * sech2 * d_inner);
}


// ------------------------------------------------------------
// Forward CUDA kernel
// ------------------------------------------------------------

__global__ void fused_bias_gelu_fwd_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int rows,
    int cols
) {
    int idx =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    int stride =
        blockDim.x * gridDim.x;

    int total = rows * cols;

    for (int i = idx; i < total; i += stride) {

        int col = i % cols;

        float val =
            x[i] + bias[col];

        y[i] = gelu_fwd(val);
    }
}


// ------------------------------------------------------------
// Backward CUDA kernel
// ------------------------------------------------------------

__global__ void fused_bias_gelu_bwd_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ grad_x,
    float* __restrict__ grad_bias,
    int rows,
    int cols
) {
    int idx =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    int stride =
        blockDim.x * gridDim.x;

    int total = rows * cols;

    for (int i = idx; i < total; i += stride) {

        int col = i % cols;

        float val =
            x[i] + bias[col];

        float local_grad =
            gelu_bwd(val) * grad_out[i];

        grad_x[i] = local_grad;

        atomicAdd(
            &grad_bias[col],
            local_grad
        );
    }
}


// ------------------------------------------------------------
// C++ wrapper: forward
// ------------------------------------------------------------

torch::Tensor fused_bias_gelu_forward(
    torch::Tensor x,
    torch::Tensor bias
) {
    TORCH_CHECK(
        x.is_cuda(),
        "x must be a CUDA tensor"
    );

    TORCH_CHECK(
        bias.is_cuda(),
        "bias must be a CUDA tensor"
    );

    TORCH_CHECK(
        x.dtype() == torch::kFloat32,
        "x must be float32"
    );

    TORCH_CHECK(
        bias.dtype() == torch::kFloat32,
        "bias must be float32"
    );

    TORCH_CHECK(
        x.dim() == 2,
        "x must be 2-dimensional"
    );

    TORCH_CHECK(
        bias.dim() == 1,
        "bias must be 1-dimensional"
    );

    TORCH_CHECK(
        x.size(1) == bias.size(0),
        "bias size must equal x.size(1)"
    );

    x = x.contiguous();
    bias = bias.contiguous();

    int rows = x.size(0);
    int cols = x.size(1);

    auto y = torch::empty_like(x);

    int threads = 256;

    int total = rows * cols;

    int blocks =
        std::min(
            (total + threads - 1) / threads,
            65535
        );

    fused_bias_gelu_fwd_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        y.data_ptr<float>(),
        rows,
        cols
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}


// ------------------------------------------------------------
// C++ wrapper: backward
// ------------------------------------------------------------

std::vector<torch::Tensor> fused_bias_gelu_backward(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor bias
) {
    grad_out = grad_out.contiguous();
    x = x.contiguous();
    bias = bias.contiguous();

    int rows = x.size(0);
    int cols = x.size(1);

    auto grad_x =
        torch::empty_like(x);

    auto grad_bias =
        torch::zeros_like(bias);

    int threads = 256;

    int total = rows * cols;

    int blocks =
        std::min(
            (total + threads - 1) / threads,
            65535
        );

    fused_bias_gelu_bwd_kernel<<<blocks, threads>>>(
        grad_out.data_ptr<float>(),
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        grad_bias.data_ptr<float>(),
        rows,
        cols
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        grad_x,
        grad_bias
    };
}


// ------------------------------------------------------------
// Python module
// ------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def(
        "forward",
        &fused_bias_gelu_forward,
        "Fused bias + GELU forward (CUDA)"
    );

    m.def(
        "backward",
        &fused_bias_gelu_backward,
        "Fused bias + GELU backward (CUDA)"
    );
}