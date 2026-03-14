// Minimal headers for PyTorch HIP extension
#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/python_variable.h>  // For THPVariable_Unpack
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <hip/hip_runtime.h>
#include <vector>

namespace py = pybind11;
using at::Tensor;

// Forward declaration of HIP kernel launcher
extern "C" void launch_triangle_kernel(
    const float* stars,
    float* scores,
    int* indices,
    int n_stars,
    int max_triangles,
    hipStream_t stream
);

// PyTorch C++ extension function - takes Python object and converts to Tensor
std::vector<Tensor> triangle_score_hip(py::object py_stars) {
    // Convert Python tensor to C++ Tensor
    Tensor stars = THPVariable_Unpack(py_stars.ptr());

    // Validate input
    TORCH_CHECK(stars.is_cuda(), "stars must be a CUDA tensor");
    TORCH_CHECK(stars.dim() == 2, "stars must be 2D");
    TORCH_CHECK(stars.size(1) == 3, "stars must have 3 columns (x, y, z)");
    TORCH_CHECK(stars.dtype() == at::kFloat, "stars must be float32");
    TORCH_CHECK(stars.is_contiguous(), "stars must be contiguous");

    int n_stars = stars.size(0);
    int max_triangles = n_stars * (n_stars - 1) * (n_stars - 2) / 6;

    // Allocate output tensors
    auto options = at::TensorOptions()
        .dtype(at::kFloat)
        .device(stars.device());

    auto scores = at::empty({max_triangles}, options);

    auto indices = at::empty({max_triangles, 3},
        options.dtype(at::kInt));

    // Use default stream (0) - simpler than getting current CUDA stream
    // which requires complex linkage against c10_hip
    hipStream_t stream = 0;

    // Launch kernel
    launch_triangle_kernel(
        stars.data_ptr<float>(),
        scores.data_ptr<float>(),
        indices.data_ptr<int>(),
        n_stars,
        max_triangles,
        stream
    );

    // Check for errors
    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    // Synchronize to catch runtime errors
    err = hipStreamSynchronize(stream);
    TORCH_CHECK(err == hipSuccess, "HIP kernel execution failed: ", hipGetErrorString(err));

    return {scores, indices};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("triangle_score", &triangle_score_hip, "Triangle scoring with HIP kernel");
}
