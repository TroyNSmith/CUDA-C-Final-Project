#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>


#ifdef __cplusplus
extern "C" {
#endif

#ifndef EXPORT
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif
#endif

#define PI 3.14159265f
/* Small epsilon to make bin assignment consistent near bin boundaries */
#define EPS 1e-6f

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                              \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

__global__ void radial_distrobution_kernel(float *coords_1, int n1,
                                            float *coords_2, int n2,
                                            float *g_r, int num_bins,
                                            float *box, float bin_width) {

        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y;
        if (row >= n1 || col >= n2)
            return;


        float dx = fabsf(coords_1[3*row + 0] - coords_2[3*col + 0]);
        float dy = fabsf(coords_1[3*row + 1] - coords_2[3*col + 1]);
        float dz = fabsf(coords_1[3*row + 2] - coords_2[3*col + 2]);

        // Apply minimum image convention
        if (dx > 0.5f * box[0]) dx = box[0] - dx;
        if (dy > 0.5f * box[1]) dy = box[1] - dy;
        if (dz > 0.5f * box[2]) dz = box[2] - dz;

        float r = sqrtf(dx*dx + dy*dy + dz*dz);

        int bin = (int)floorf(r / bin_width + 1e-6f);

        if (bin >= 0 && bin < num_bins)
            atomicAdd(&g_r[bin], 1.0f);
}

EXPORT double radial_distribution_cuda(
    const float *coords_1, int n1,
    const float *coords_2, int n2,
    float *g_r, int num_bins,
    const float *box, float r_max)
{
    if (!coords_1 || !coords_2 || !g_r || !box) {
        fprintf(stderr, "[Error] Null pointer in arguments\n");
        return -1;
    }

    float bin_width = r_max / num_bins;


    float *d_coords_1;
    float *d_coords_2;
    float *d_box;
    float *d_g_r;
    CUDA_CHECK(cudaMalloc((void**)&d_coords_1, n1 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_coords_2, n2 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_box, 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_g_r, num_bins * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_r, 0, num_bins * sizeof(float)));


    CUDA_CHECK(cudaMemcpy(d_coords_1, coords_1, n1 * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coords_2, coords_2, n2 * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_box, box, 3 * sizeof(float), cudaMemcpyHostToDevice));


    auto start = std::chrono::high_resolution_clock::now(); // Start time
    // Use a block size that keeps threads-per-block <= 1024 (e.g., 32x32 = 1024)
    dim3 blockSize(32, 32, 1);
    dim3 gridSize((n1 + blockSize.x - 1) / blockSize.x, (n2 + blockSize.y - 1) / blockSize.y, 1);
    radial_distrobution_kernel<<<gridSize, blockSize>>>(
        d_coords_1, n1,
        d_coords_2, n2,
        d_g_r, num_bins,
        d_box, bin_width
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();   // End time

    std::chrono::duration<double> elapsed = end - start;    // Calculate duration
    double time = elapsed.count();

    CUDA_CHECK(cudaMemcpy(g_r, d_g_r, num_bins * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_coords_1));
    CUDA_CHECK(cudaFree(d_coords_2));
    CUDA_CHECK(cudaFree(d_box));
    CUDA_CHECK(cudaFree(d_g_r));
    return time;
}

#ifdef __cplusplus
}
#endif


