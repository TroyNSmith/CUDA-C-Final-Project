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

__global__ void localSMKernel(
    float *A, int n,
    float *B, int m,
    float *g_r, int num_bins,
    float *box, float bin_width) {
    
    extern __shared__ unsigned int local_hist[];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threads = blockDim.x * blockDim.y;
    
    // Use a local histogram to do atomic adds in SM rather than GM
    for (int i = tid; i < num_bins; i += threads)
        local_hist[i] = 0.0f;
    __syncthreads();
    
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= m) return;

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
        atomicAdd(&local_hist[bin], 1.0f);
    __syncthreads();
    
    // GM atomic adds once per bin rather than once per thread
    for (int i = tid; i < num_bins; i += threads)
        if (local_hist[i] > 0)
            atomicAdd(&g_r[i], local_hist[i]);
}

/* Constant memory holds one tile of B (up to BLOCK_SIZE atoms, 3 coords each) */
__constant__ float B_C[65536 / sizeof(float)];

__global__ void tunedTiledJoshCudaKernel(
    float *A, int n, int m,
    float *g_r, int num_bins, float bin_width,
    float box_x, float box_y, float box_z, int tile_y, int iterations, int number_of_tiles_y) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col_in_tile = (blockIdx.y * blockDim.y * iterations) + threadIdx.y;
    int col = tile_y * (170 * blockDim.y) + col_in_tile;

    if (row >= n || col >= m || col > row)
        return;

    float Ax = A[3*row + 0];
    float Ay = A[3*row + 1];
    float Az = A[3*row + 2];

    for (int i = 0; i < iterations; i++) {
        if ((col + 32 * i >= m) || (col + 32 * i > row))
            return;
        float dx = fabsf(Ax - B_C[3*(col_in_tile + 32 * i) + 0]);
        float dy = fabsf(Ay - B_C[3*(col_in_tile + 32 * i) + 1]);
        float dz = fabsf(Az - B_C[3*(col_in_tile + 32 * i) + 2]);

        // Apply minimum image convention
        if (dx > 0.5f * box_x) dx = box_x - dx;
        if (dy > 0.5f * box_y) dy = box_y - dy;
        if (dz > 0.5f * box_z) dz = box_z - dz;

        float r = sqrtf(dx*dx + dy*dy + dz*dz);

        int bin = (int)floorf(r / bin_width + EPS);

        // if (tile_y != 0) printf("Row %d Col %d bin = %d\n", row, col_in_tile + 32 * i, bin);

        if (bin > 0 && bin < num_bins)
            atomicAdd(&g_r[bin], 2.0f);
    }
}

__global__ void tiledJoshCudaKernel(
    float *A, int n, int m,
    float *g_r, int num_bins, float bin_width,
    float box_x, float box_y, float box_z, int tile_y, int number_of_tiles_y) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col_in_tile = threadIdx.y;
    int col = tile_y * (170 * blockDim.y) + col_in_tile;

    if (row >= n || col >= m || col > row)
        return;

    float Ax = A[3*row + 0];
    float Ay = A[3*row + 1];
    float Az = A[3*row + 2];

    for (int i = 0; i < number_of_tiles_y; i++) {
        if ((col + 32 * i >= m) || (col + 32 * i > row))
            return;
        float dx = fabsf(Ax - B_C[3*(col_in_tile + 32 * i) + 0]);
        float dy = fabsf(Ay - B_C[3*(col_in_tile + 32 * i) + 1]);
        float dz = fabsf(Az - B_C[3*(col_in_tile + 32 * i) + 2]);

        // Apply minimum image convention
        if (dx > 0.5f * box_x) dx = box_x - dx;
        if (dy > 0.5f * box_y) dy = box_y - dy;
        if (dz > 0.5f * box_z) dz = box_z - dz;

        float r = sqrtf(dx*dx + dy*dy + dz*dz);

        int bin = (int)floorf(r / bin_width + EPS);

        // if (tile_y != 0) printf("Row %d Col %d bin = %d\n", row, col_in_tile + 32 * i, bin);

        if (bin > 0 && bin < num_bins)
            atomicAdd(&g_r[bin], 2.0f);
    }
}

__global__ void joshCudaKernel(
    float *A, int n, int m,
    float *g_r, int num_bins, float bin_width,
    float box_x, float box_y, float box_z, int tile_y) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col_in_tile = blockIdx.y * blockDim.y + threadIdx.y;
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

    //if (tile_y != 0) printf("Row %d Col %d bin = %d\n", row, col_in_tile, bin);

    if (bin > 0 && bin < num_bins)
        atomicAdd(&g_r[bin], 2.0f);
}

__global__ void tiledLocalSMKernel(
    float *A, int n, int m,
    float *g_r, int num_bins, float bin_width,
    float box_x, float box_y, float box_z, int tile_y) {

    // Thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= n) return; // out-of-bounds

    // Global column index in B
    int col = tile_y * (blockDim.y * gridDim.y) + y;
    if (col >= m) return; // out-of-bounds column

    // Linear thread index in block
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threads = blockDim.x * blockDim.y;

    // Shared-memory histogram per block
    extern __shared__ unsigned int local_hist[];
    for (int i = tid; i < num_bins; i += threads)
        local_hist[i] = 0;
    __syncthreads();

    float bx = B_C[3*y + 0];
    float by = B_C[3*y + 1];
    float bz = B_C[3*y + 2];

    float dx = fabsf(A[3*x + 0] - bx);
    float dy = fabsf(A[3*x + 1] - by);
    float dz = fabsf(A[3*x + 2] - bz);

    // Minimum image
    if (dx > 0.5f * box_x) dx = box_x - dx;
    if (dy > 0.5f * box_y) dy = box_y - dy;
    if (dz > 0.5f * box_z) dz = box_z - dz;

    float r = sqrtf(dx*dx + dy*dy + dz*dz);

    int bin = (int)floorf(r / bin_width + EPS);
    if (bin > 0 && bin < num_bins)
        atomicAdd(&local_hist[bin], 1);

    __syncthreads();

    // Reduce local histogram to global histogram
    for (int i = tid; i < num_bins; i += threads)
        if (local_hist[i] > 0)
            atomicAdd(&g_r[i], local_hist[i]);
}

#ifdef __cplusplus
}
#endif


