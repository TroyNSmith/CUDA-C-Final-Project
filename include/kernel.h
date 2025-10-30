#ifndef KERNEL_H
#define KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

void naiveKernel(float *A, int n,
                 float *B, int m,
                 float *g_r, int num_bins,
                 float *box, float bin_width);

__global__ void cudaKernel(float *A, int n,
                           float *B, int m,
                           float *g_r, int num_bins,
                           float *box, float bin_width);

#ifdef __cplusplus
}
#endif

#endif