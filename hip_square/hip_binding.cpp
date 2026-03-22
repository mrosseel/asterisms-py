#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <limits>

namespace py = pybind11;
using at::Tensor;

extern "C" void launch_square_kernel(
    const float* stars,
    float* scores,
    int* indices,
    int n_stars,
    long long max_quads,
    float max_dist,
    hipStream_t stream
);

std::vector<Tensor> square_score_hip(py::object py_stars, float max_dist = std::numeric_limits<float>::infinity()) {
    Tensor stars = THPVariable_Unpack(py_stars.ptr());

    TORCH_CHECK(stars.is_cuda(), "stars must be a CUDA tensor");
    TORCH_CHECK(stars.dim() == 2, "stars must be 2D");
    TORCH_CHECK(stars.size(1) == 3, "stars must have 3 columns (x, y, z)");
    TORCH_CHECK(stars.dtype() == at::kFloat, "stars must be float32");
    TORCH_CHECK(stars.is_contiguous(), "stars must be contiguous");

    int n_stars = stars.size(0);
    long long max_quads = (long long)n_stars * (n_stars - 1) * (n_stars - 2) * (n_stars - 3) / 24;

    auto options = at::TensorOptions()
        .dtype(at::kFloat)
        .device(stars.device());

    auto scores = at::empty({max_quads}, options);
    auto indices = at::empty({max_quads, 4}, options.dtype(at::kInt));

    hipStream_t stream = 0;

    launch_square_kernel(
        stars.data_ptr<float>(),
        scores.data_ptr<float>(),
        indices.data_ptr<int>(),
        n_stars,
        max_quads,
        max_dist,
        stream
    );

    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    err = hipStreamSynchronize(stream);
    TORCH_CHECK(err == hipSuccess, "HIP kernel execution failed: ", hipGetErrorString(err));

    return {scores, indices};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("square_score", &square_score_hip, "Square scoring with HIP kernel",
          py::arg("stars"), py::arg("max_dist") = std::numeric_limits<float>::infinity());
}
