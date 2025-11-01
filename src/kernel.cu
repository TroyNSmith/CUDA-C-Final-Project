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

/* Small epsilon to make bin assignment consistent near bin boundaries */
#define EPS 1e-6f

#define BLOCK_SIZE 32

void naiveKernel(
    const float *A, int n,
    const float *B, int m,
    float *g_r, int num_bins,
    const float *box, float r_max) {

    if (!A || !B || !g_r || !box) {
        return;
    }

    float bin_width = r_max / num_bins;

    // Zero the histogram
    for (int i = 0; i < num_bins; i++)
        g_r[i] = 0.0f;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {

            float dx = fabsf(A[3*i + 0] - B[3*j + 0]);
            float dy = fabsf(A[3*i + 1] - B[3*j + 1]);
            float dz = fabsf(A[3*i + 2] - B[3*j + 2]);

            // Apply minimum image convention
            if (dx > 0.5f * box[0]) dx = box[0] - dx;
            if (dy > 0.5f * box[1]) dy = box[1] - dy;
            if (dz > 0.5f * box[2]) dz = box[2] - dz;

            float r = sqrtf(dx*dx + dy*dy + dz*dz);
                
            int bin = (int)floorf(r / bin_width + EPS);

            if (bin > 0 && bin < num_bins)
                g_r[bin] += 1.0f;
        }
    }
}

__global__ void cudaKernel(
    float *A, int n,
    float *B, int m,
    float *g_r, int num_bins,
    float *box, float bin_width) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= m)
        return;

    float dx = fabsf(A[3*row + 0] - B[3*col + 0]);
    float dy = fabsf(A[3*row + 1] - B[3*col + 1]);
    float dz = fabsf(A[3*row + 2] - B[3*col + 2]);

    // Apply minimum image convention
    if (dx > 0.5f * box[0]) dx = box[0] - dx;
    if (dy > 0.5f * box[1]) dy = box[1] - dy;
    if (dz > 0.5f * box[2]) dz = box[2] - dz;

    float r = sqrtf(dx*dx + dy*dy + dz*dz);

    int bin = (int)floorf(r / bin_width + EPS);

    if (bin > 0 && bin < num_bins)
        atomicAdd(&g_r[bin], 1.0f);
}

/* Constant memory holds one tile of B (up to BLOCK_SIZE atoms, 3 coords each) */
__constant__ float B_C[65536 / sizeof(float)];

__global__ void joshCudaKernel(
    float *A, int n,
    int m,
    float *g_r, int num_bins,
    float bin_width, float box_x, float box_y, float box_z,
    int tile_y) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col_in_tile = blockDim.y * blockIdx.y + threadIdx.y;
    int col = tile_y * (170 * blockDim.y) + col_in_tile;

    if (row >= n || col >= m || col > row)
        return;

    //printf("Row: %d, Col: %d, Tile: %d, col_in_tile = %d\n", threadIdx.x, threadIdx.y, tile_y, col_in_tile);

    float dx = fabsf(A[3*row + 0] - B_C[3*col_in_tile + 0]);
    float dy = fabsf(A[3*row + 1] - B_C[3*col_in_tile + 1]);
    float dz = fabsf(A[3*row + 2] - B_C[3*col_in_tile + 2]);

    // Apply minimum image convention
    if (dx > 0.5f * box_x) dx = box_x - dx;
    if (dy > 0.5f * box_y) dy = box_y - dy;
    if (dz > 0.5f * box_z) dz = box_z - dz;

    float r = sqrtf(dx*dx + dy*dy + dz*dz);

    int bin = (int)floorf(r / bin_width + EPS);

    if (bin > 0 && bin < num_bins)
        atomicAdd(&g_r[bin], 2.0f);
}

// EXPORT double radial_distribution_cuda(
//     const float *coords_1, int n1,
//     const float *coords_2, int n2,
//     float *g_r, int num_bins,
//     const float *box, float r_max)
// {
//     if (!coords_1 || !coords_2 || !g_r || !box) {
//         fprintf(stderr, "[Error] Null pointer in arguments\n");
//         return -1;
//     }

//     float bin_width = r_max / num_bins;

//     float *d_coords_1;
//     float *d_coords_2;
//     float *d_box;
//     float *d_g_r;
//     CUDA_CHECK(cudaMalloc((void**)&d_coords_1, n1 * 3 * sizeof(float)));
//     CUDA_CHECK(cudaMalloc((void**)&d_coords_2, n2 * 3 * sizeof(float)));
//     CUDA_CHECK(cudaMalloc((void**)&d_box, 3 * sizeof(float)));
//     CUDA_CHECK(cudaMalloc((void**)&d_g_r, num_bins * sizeof(float)));
//     CUDA_CHECK(cudaMemset(d_g_r, 0, num_bins * sizeof(float)));


//     CUDA_CHECK(cudaMemcpy(d_coords_1, coords_1, n1 * 3 * sizeof(float), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_coords_2, coords_2, n2 * 3 * sizeof(float), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_box, box, 3 * sizeof(float), cudaMemcpyHostToDevice));


//     auto start = std::chrono::high_resolution_clock::now(); // Start time
//     // Use a block size that keeps threads-per-block <= 1024 (e.g., 32x32 = 1024)
//     dim3 blockSize(32, 32, 1);
//     dim3 gridSize((n1 + blockSize.x - 1) / blockSize.x, (n2 + blockSize.y - 1) / blockSize.y, 1);
//     radial_distribution_kernel<<<gridSize, blockSize>>>(
//         d_coords_1, n1,
//         d_coords_2, n2,
//         d_g_r, num_bins,
//         d_box, bin_width
//     );

//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());
//     auto end = std::chrono::high_resolution_clock::now();   // End time

//     std::chrono::duration<double> elapsed = end - start;    // Calculate duration
//     double time = elapsed.count();

//     CUDA_CHECK(cudaMemcpy(g_r, d_g_r, num_bins * sizeof(float), cudaMemcpyDeviceToHost));

//     CUDA_CHECK(cudaFree(d_coords_1));
//     CUDA_CHECK(cudaFree(d_coords_2));
//     CUDA_CHECK(cudaFree(d_box));
//     CUDA_CHECK(cudaFree(d_g_r));
//     return time;
// }

#ifdef __cplusplus
}
#endif


