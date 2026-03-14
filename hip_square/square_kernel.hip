#include <hip/hip_runtime.h>
#include <math.h>

// Score a single quadrilateral cycle: 4 sides + 2 diagonals → squareness score
__device__ float score_cycle(
    float d01, float d02, float d03, float d12, float d13, float d23,
    // Which of the 6 distances are sides (4) and diags (2):
    float s0, float s1, float s2, float s3, float dg0, float dg1
) {
    float mean_s = (s0 + s1 + s2 + s3) * 0.25f;
    float mean_d = (dg0 + dg1) * 0.5f;

    // CV of sides
    float var_s = ((s0 - mean_s)*(s0 - mean_s) + (s1 - mean_s)*(s1 - mean_s) +
                   (s2 - mean_s)*(s2 - mean_s) + (s3 - mean_s)*(s3 - mean_s)) * 0.25f;
    float cv_s = (mean_s > 1e-8f) ? sqrtf(var_s) / mean_s : 1.0f;

    // CV of diagonals
    float var_d = ((dg0 - mean_d)*(dg0 - mean_d) + (dg1 - mean_d)*(dg1 - mean_d)) * 0.5f;
    float cv_d = (mean_d > 1e-8f) ? sqrtf(var_d) / mean_d : 1.0f;

    // Ratio deviation from sqrt(2)
    float ratio_dev = fabsf(mean_d / fmaxf(mean_s, 1e-8f) - 1.4142135f);

    return cv_s + cv_d + ratio_dev;
}


__global__ void square_score_hip_kernel(
    const float* __restrict__ stars,
    float* __restrict__ scores,
    int* __restrict__ indices,
    int n_stars,
    long long max_quads
) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= max_quads) return;

    // Combinatorial unranking: linear index → (a, b, c, d) where a < b < c < d
    long long remaining = tid;
    int a = 0;

    // Find a: skip C(n-a-1, 3) at a time
    while (true) {
        long long n_rem = n_stars - a - 1;
        long long combos_a = n_rem * (n_rem - 1) * (n_rem - 2) / 6;
        if (remaining < combos_a) break;
        remaining -= combos_a;
        a++;
    }

    // Find b: skip C(n-b-1, 2) at a time
    int b = a + 1;
    while (true) {
        long long n_rem = n_stars - b - 1;
        long long combos_b = n_rem * (n_rem - 1) / 2;
        if (remaining < combos_b) break;
        remaining -= combos_b;
        b++;
    }

    // Find c: skip (n-c-1) at a time
    int c = b + 1;
    while (true) {
        int combos_c = n_stars - c - 1;
        if (remaining < combos_c) break;
        remaining -= combos_c;
        c++;
    }

    int d = c + 1 + (int)remaining;

    if (a >= n_stars || b >= n_stars || c >= n_stars || d >= n_stars) return;

    // Load 4 star positions
    float ax_ = stars[a*3+0], ay = stars[a*3+1], az = stars[a*3+2];
    float bx  = stars[b*3+0], by = stars[b*3+1], bz = stars[b*3+2];
    float cx  = stars[c*3+0], cy = stars[c*3+1], cz = stars[c*3+2];
    float dx_ = stars[d*3+0], dy = stars[d*3+1], dz = stars[d*3+2];

    // Compute all 6 pairwise distances
    float ex, ey, ez;

    ex = bx-ax_; ey = by-ay; ez = bz-az; float d01 = sqrtf(ex*ex+ey*ey+ez*ez);
    ex = cx-ax_; ey = cy-ay; ez = cz-az; float d02 = sqrtf(ex*ex+ey*ey+ez*ez);
    ex = dx_-ax_; ey = dy-ay; ez = dz-az; float d03 = sqrtf(ex*ex+ey*ey+ez*ez);
    ex = cx-bx; ey = cy-by; ez = cz-bz; float d12 = sqrtf(ex*ex+ey*ey+ez*ez);
    ex = dx_-bx; ey = dy-by; ez = dz-bz; float d13 = sqrtf(ex*ex+ey*ey+ez*ez);
    ex = dx_-cx; ey = dy-cy; ez = dz-cz; float d23 = sqrtf(ex*ex+ey*ey+ez*ez);

    // Test all 3 quadrilateral cycles, pick best score
    // Cycle A-B-C-D: sides AB,BC,CD,DA; diags AC,BD
    float s1 = score_cycle(d01, d02, d03, d12, d13, d23,
                           d01, d12, d23, d03, d02, d13);
    // Cycle A-B-D-C: sides AB,BD,DC,CA; diags AD,BC
    float s2 = score_cycle(d01, d02, d03, d12, d13, d23,
                           d01, d13, d23, d02, d03, d12);
    // Cycle A-C-B-D: sides AC,CB,BD,DA; diags AB,CD
    float s3 = score_cycle(d01, d02, d03, d12, d13, d23,
                           d02, d12, d13, d03, d01, d23);

    float best = fminf(s1, fminf(s2, s3));

    scores[tid] = best;
    indices[tid * 4 + 0] = a;
    indices[tid * 4 + 1] = b;
    indices[tid * 4 + 2] = c;
    indices[tid * 4 + 3] = d;
}


extern "C" void launch_square_kernel(
    const float* stars,
    float* scores,
    int* indices,
    int n_stars,
    long long max_quads,
    hipStream_t stream
) {
    int threads_per_block = 256;
    long long blocks = (max_quads + threads_per_block - 1) / threads_per_block;

    hipLaunchKernelGGL(
        square_score_hip_kernel,
        dim3((unsigned int)blocks),
        dim3(threads_per_block),
        0,
        stream,
        stars, scores, indices, n_stars, max_quads
    );
}
