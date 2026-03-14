#include <hip/hip_runtime.h>
#include <math.h>

// HIP kernel for parallel triangle generation and scoring
__global__ void triangle_score_hip_kernel(
    const float* __restrict__ stars,
    float* __restrict__ scores,
    int* __restrict__ indices,
    int n_stars,
    int max_triangles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= max_triangles) return;

    // Combinatorial unranking: convert linear thread ID to (i, j, k)
    // This avoids nested loops and makes it fully parallel
    int remaining = tid;
    int i = 0;

    // Find i
    while (true) {
        int combos_with_i = (n_stars - i - 1) * (n_stars - i - 2) / 2;
        if (remaining < combos_with_i) break;
        remaining -= combos_with_i;
        i++;
    }

    // Find j
    int j = i + 1;
    while (true) {
        int combos_with_j = n_stars - j - 1;
        if (remaining < combos_with_j) break;
        remaining -= combos_with_j;
        j++;
    }

    // k is whatever's left
    int k = j + 1 + remaining;

    // Bounds check (shouldn't happen but safety first)
    if (i >= n_stars || j >= n_stars || k >= n_stars) return;

    // Get star positions (stars is [n_stars, 3])
    float p1_x = stars[i * 3 + 0];
    float p1_y = stars[i * 3 + 1];
    float p1_z = stars[i * 3 + 2];

    float p2_x = stars[j * 3 + 0];
    float p2_y = stars[j * 3 + 1];
    float p2_z = stars[j * 3 + 2];

    float p3_x = stars[k * 3 + 0];
    float p3_y = stars[k * 3 + 1];
    float p3_z = stars[k * 3 + 2];

    // Calculate side lengths
    float dx = p2_x - p1_x;
    float dy = p2_y - p1_y;
    float dz = p2_z - p1_z;
    float a = sqrtf(dx*dx + dy*dy + dz*dz);

    dx = p3_x - p2_x;
    dy = p3_y - p2_y;
    dz = p3_z - p2_z;
    float b = sqrtf(dx*dx + dy*dy + dz*dz);

    dx = p1_x - p3_x;
    dy = p1_y - p3_y;
    dz = p1_z - p3_z;
    float c = sqrtf(dx*dx + dy*dy + dz*dz);

    // Raw coefficient of variation (scale-invariant, no normalization needed)
    float mean = (a + b + c) / 3.0f;
    float variance = ((a - mean)*(a - mean) + (b - mean)*(b - mean) + (c - mean)*(c - mean)) / 3.0f;
    float std_dev = sqrtf(variance);

    float score = (mean > 1e-8f) ? (std_dev / mean) : 1.0f;

    // Store results
    scores[tid] = score;
    indices[tid * 3 + 0] = i;
    indices[tid * 3 + 1] = j;
    indices[tid * 3 + 2] = k;
}

// Host function to launch kernel
extern "C" void launch_triangle_kernel(
    const float* stars,
    float* scores,
    int* indices,
    int n_stars,
    int max_triangles,
    hipStream_t stream
) {
    int threads_per_block = 256;
    int blocks = (max_triangles + threads_per_block - 1) / threads_per_block;

    hipLaunchKernelGGL(
        triangle_score_hip_kernel,
        dim3(blocks),
        dim3(threads_per_block),
        0,  // shared memory
        stream,
        stars, scores, indices, n_stars, max_triangles
    );
}
