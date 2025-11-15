#ifndef KERNEL_H
#define KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

void naiveKernel(
    float *A, int n,
    float *B, int m,
    float *g_r, int num_bins,
    float *box, float bin_width);

__global__ void cudaKernel(
    float *A, int n,
    float *B, int m,
    float *g_r, int num_bins,
    float *box, float bin_width);

__global__ void localSMKernel(float *A, int n,
                              float *B, int m,
                              float *g_r, int num_bins,
                              float *box, float bin_width);

__global__ void tiledLocalSMKernel(float *A, int n, int m,
                                   float *g_r, int num_bins,
                                   float bin_width, float box_x, float box_y, float box_z);

__global__ void joshCudaKernel(float *A, int n, int m,
                               float *g_r, int num_bins,
                               float bin_width, float box_x, float box_y, float box_z, int tile_y);

__global__ void tiledJoshCudaKernel(
    float *A, int n, int m,
    float *g_r, int num_bins, float bin_width,
    float box_x, float box_y, float box_z, int tile_y, int number_of_tiles_y);

#ifdef __cplusplus
}
#endif

#endif