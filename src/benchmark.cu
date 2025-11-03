#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

#include "support.h"
#include "kernel.cu"

// Structure for simulated data (to use with kernelRun)
struct SimulationData {
    float *A_d, *B_d, *G_d, *Box_d;
    float *A_h, *B_h, *G_h, *G_reference;
	float bin_width;
    unsigned int n, m;
};
SimulationData sim;

// Verify integrity of results
void verify(float *A, float *B, int n)
{
    const float Tolerance = 0.05 * n;
    for (int i = 0; i < n; i++) {
        float diff = A[i] - B[i];
        if (diff > Tolerance || diff < -Tolerance) {
            printf("\nEntry %d differs: GPU = %f, CPU = %f, diff = %f\n",
                   i, A[i], B[i], diff);
            printf("TEST FAILED\n\n");
            // exit(EXIT_FAILURE);
			return;
        }
    }
    printf("\nTEST PASSED\n\n");
}

// Kernel runner to make things less chaotic
void runKernel(const char *label, void (*launchFunc)(void),
               float *G_h, float *G_d, float *G_reference,
               int num_bins)
{
    Timer timer;
    printf("\nLaunching %s...", label);
    fflush(stdout);

    cudaMemset(G_d, 0, num_bins * sizeof(float));

    // --- Launch & time kernel ---
    startTime(&timer);
    launchFunc();
    cudaError_t cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess) {
        fprintf(stderr, "\nError launching %s: %s\n",
                label, cudaGetErrorString(cuda_ret));
        exit(EXIT_FAILURE);
    }
    stopTime(&timer);
    printf(" %f s\n", elapsedTime(timer));

    // --- Copy back results ---
    printf("Copying results from device to host...");
    fflush(stdout);
    startTime(&timer);
    cudaMemcpy(G_h, G_d, num_bins * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printf(" %f s\n", elapsedTime(timer));

    // --- Verify ---
    printf("Verifying results...");
    fflush(stdout);
    verify(G_h, G_reference, num_bins);
}

// Kernel launchers
#define r_max      2.50
#define num_bins   1000
#define box_size   10.0
#define BLOCK_SIZE 32

void launchCudaKernel(void) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((sim.n + blockSize.x - 1) / blockSize.x,
                  (sim.m + blockSize.y - 1) / blockSize.y);

    cudaKernel<<<gridSize, blockSize>>>(
        sim.A_d, sim.n, sim.B_d, sim.m,
        sim.G_d, num_bins, sim.Box_d, r_max / num_bins
    );
}

void launchLocalSMKernel(void) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((sim.n + blockSize.x - 1) / blockSize.x,
                  (sim.m + blockSize.y - 1) / blockSize.y);

    size_t shmem = num_bins * sizeof(float);
    localSMKernel<<<gridSize, blockSize, shmem>>>(
        sim.A_d, sim.n, sim.B_d, sim.m,
        sim.G_d, num_bins, sim.Box_d, r_max / num_bins
    );
}

void launchTiledLocalSMKernel(void) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    int atomsInConstantMemory = 170 * blockSize.y;
    int tilesY = (sim.m + atomsInConstantMemory - 1) / atomsInConstantMemory;
	size_t shmem = num_bins * sizeof(float);

    for (int tile = 0; tile < tilesY; ++tile) {
        int colsInTile = (tile == tilesY - 1)
                             ? sim.m - tile * atomsInConstantMemory
                             : atomsInConstantMemory;
		
        if (colsInTile <= 0) break;

        cudaMemcpyToSymbol(B_C, &sim.B_h[tile * atomsInConstantMemory * 3],
            			   colsInTile * 3 * sizeof(float), 0, cudaMemcpyHostToDevice);

        dim3 tileGrid((sim.n + blockSize.x - 1) / blockSize.x,
                      (colsInTile + blockSize.y - 1) / blockSize.y);

        tiledLocalSMKernel<<<tileGrid, blockSize, shmem>>>(
            sim.A_d, sim.n, sim.m, sim.G_d,
            num_bins, r_max / num_bins,
            box_size, box_size, box_size, tile
        );
    }
}

void launchJoshKernel(void) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    int atomsInConstantMemory = 170 * blockSize.y;
    int tilesY = (sim.m + atomsInConstantMemory - 1) / atomsInConstantMemory;

    for (int tile = 0; tile < tilesY; ++tile) {
        int colsInTile = (tile == tilesY - 1)
                             ? sim.m - tile * atomsInConstantMemory
                             : atomsInConstantMemory;
        if (colsInTile <= 0) break;

        cudaMemcpyToSymbol(B_C, &sim.B_h[tile * atomsInConstantMemory * 3],
            			   colsInTile * 3 * sizeof(float), 0, cudaMemcpyHostToDevice);

        dim3 tileGrid((sim.n + blockSize.x - 1) / blockSize.x,
                      (colsInTile + blockSize.y - 1) / blockSize.y);

        joshCudaKernel<<<tileGrid, blockSize>>>(
            sim.A_d, sim.n, sim.m, sim.G_d,
            num_bins, r_max / num_bins,
            box_size, box_size, box_size, tile
        );
    }
}

// Main logic
int main(int argc, char **argv)
{
    // --- Device info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Total constant memory: %d bytes\n", (int)deviceProp.totalConstMem);

    Timer timer;

    // --- Parse arguments
    unsigned int n, m, distinct;
    if (argc == 1) {
        n = m = 10000; distinct = 0;
    } else if (argc == 2) {
        n = m = atoi(argv[1]); distinct = 0;
    } else if (argc == 3) {
        n = atoi(argv[1]); m = atoi(argv[2]); distinct = 1;
    } else {
        printf("\nUsage: ./benchmark [n] [m]\n");
        return EXIT_FAILURE;
    }

    // --- Host data setup
    printf("\nSetting up the problem...");
    fflush(stdout);
    startTime(&timer);

	float bin_width = r_max / num_bins;

    float *A_h = (float*) malloc(3 * n * sizeof(float));
    generateCoordinates(A_h, n, box_size);

    float *B_h = (float*) malloc(3 * m * sizeof(float));
    if (distinct) generateCoordinates(B_h, m, box_size);
    else memcpy(B_h, A_h, 3 * m * sizeof(float));

    float *G_h = (float*) calloc(num_bins, sizeof(float));
    float Box_h[3] = { box_size, box_size, box_size };

    stopTime(&timer);
    printf(" %f s\n", elapsedTime(timer));

    // --- Device allocations
    printf("\nAllocating device memory...");
    fflush(stdout);
    startTime(&timer);

    float *A_d, *B_d, *G_d, *Box_d, *EdgesSq_d;
    cudaMalloc((void**)&A_d, 3 * n * sizeof(float));
    cudaMalloc((void**)&B_d, 3 * m * sizeof(float));
    cudaMalloc((void**)&G_d, num_bins * sizeof(float));
    cudaMalloc((void**)&Box_d, 3 * sizeof(float));

    stopTime(&timer);
    printf(" %f s\n", elapsedTime(timer));

    // --- Copy host to device
    printf("\nCopying data to device...");
    fflush(stdout);
    startTime(&timer);

    cudaMemcpy(A_d, A_h, 3 * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, 3 * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(G_d, G_h, num_bins * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Box_d, Box_h, 3 * sizeof(float), cudaMemcpyHostToDevice);

    stopTime(&timer);
    printf(" %f s\n", elapsedTime(timer));

    // --- Naive CPU reference
    printf("\nLaunching naive kernel...");
    fflush(stdout);
    startTime(&timer);
    naiveKernel(A_h, n, B_h, m, G_h, num_bins, Box_h, r_max);
    stopTime(&timer);
    printf(" %f s\n", elapsedTime(timer));

    // --- Copy reference
    float *G_reference = (float*) malloc(num_bins * sizeof(float));
    memcpy(G_reference, G_h, num_bins * sizeof(float));

    // --- Store simulation data globally
    sim = { A_d, B_d, G_d, Box_d, A_h, B_h, G_h, G_reference, bin_width, n, m };

    // --- Run all kernels
    runKernel("Original CUDA kernel", launchCudaKernel, G_h, G_d, G_reference, num_bins);
    runKernel("Local SM kernel",  launchLocalSMKernel, G_h, G_d, G_reference, num_bins);
    runKernel("Tiled kernel",  launchJoshKernel, G_h, G_d, G_reference, num_bins);
	runKernel("Tiled Local SM kernel",  launchTiledLocalSMKernel, G_h, G_d, G_reference, num_bins);

    // --- Cleanup
    free(A_h); free(B_h); free(G_h); free(G_reference);
    cudaFree(A_d); cudaFree(B_d); cudaFree(G_d); cudaFree(Box_d);

    return 0;
}