#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

#include "support.h"
#include "kernel.h"


#ifdef __cplusplus
extern "C" {
#endif

void verify(float *A, float *B, int n)
{
	const float Tolerance = 1e-2;
	for (int i = 0; i < n; i++)
	{
		float difference = A[i] - B[i];
		if (difference > Tolerance ||
			difference < -Tolerance)
		{
			printf("\nTEST FAILED\n\n");
			exit(0);
		}
	}

	printf("\nTEST PASSED\n\n");
}

#define r_max 2.50
#define num_bins 1000
#define box_size 10.0

int main(int argc, char **argv)
{
	Timer timer;
	cudaError_t cuda_ret;

	// Initialize host variables
	printf("\nSetting up the problem...");
	fflush(stdout);
	startTime(&timer);

	unsigned int n; unsigned int m;
    unsigned int distinct;
	if (argc == 1) {
		n = 10000; m = 10000;
        distinct = 0; // If the atoms are not distinct, we don't need to include the lower triangle.
					  // Also, the histograms will be twice as high if we don't correct for it (either divide out or don't include lower triangle)
    }

	else if (argc == 2) {
		n = atoi(argv[1]); m = atoi(argv[1]);
		distinct = 0;
    }

	else if (argc == 3) {
		n = atoi(argv[1]); m = atoi(argv[2]);
        distinct = 1;
    }

	else {
		printf(
			"\nInvalid input parameters."
			"\n     Usage:  ./benchmark          # RDF is calculated for 10,000 atoms to itself"
            "\n             ./benchmark <n>      # RDF is calculated for n atoms to itself"
            "\n             ./benchmark <n> <m>  # RDF is calculated for n atoms to m atoms\n"
		);
	}

	float *A_h = (float*) malloc(3 * n * sizeof(float));
	generateCoordinates(A_h, n, box_size);

	float *B_h;
	if (distinct) {
		B_h = (float*) malloc(3 * m * sizeof(float));
		generateCoordinates(B_h, m, box_size);
	}
	else {
		B_h = (float*) malloc(3 * m * sizeof(float));
		memcpy(B_h, A_h, 3 * m * sizeof(float));
	}

	float *G_h = (float*) malloc(num_bins * sizeof(float));
	float Box_h[3] = {box_size, box_size, box_size};

	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));

	// Allocate device variables
	printf("\nAllocating device variables...");
	fflush(stdout);
	startTime(&timer);	

	float *A_d;
	cudaMalloc((void **) &A_d, 3 * n * sizeof(float));

	float *B_d;
	cudaMalloc((void **) &B_d, 3 * m * sizeof(float));

	float *G_d;
	cudaMalloc((void **) &G_d, num_bins * sizeof(float));

	float *Box_d;
	cudaMalloc((void **) &Box_d, 3 * sizeof(float));

	cudaDeviceSynchronize();
	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));

	// Copy host data to device
	printf("\nCopying data from host to device...");
	fflush(stdout);
	startTime(&timer);	

	cudaMemcpy(A_d, A_h, 3 * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, 3 * m * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(G_d, G_h, num_bins * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Box_d, Box_h, 3 * sizeof(float), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));	

	// Launch naive kernel
	printf("\nLaunching naive kernel...");
	fflush(stdout);
	startTime(&timer);

	naiveKernel(A_h, n, B_h, m, G_h, num_bins, Box_h, r_max);

	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));

	// Establish reference G from naive kernel
	float *G_reference = (float*) malloc(num_bins * sizeof(float));
	memcpy(G_reference, G_h, num_bins * sizeof(float));

	// Launch CUDA kernel
	cudaMemset(G_d, 0, num_bins * sizeof(float));

	printf("\nLaunching CUDA kernel...");
	fflush(stdout);
	startTime(&timer);
	
	dim3 blockSize(32, 32);
	dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);

	cudaKernel<<<gridSize, blockSize>>>(A_d, n, B_d, m, G_d, num_bins, Box_d, r_max / num_bins);

	cuda_ret = cudaDeviceSynchronize();
	if (cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));

	// Copy device data to host
	printf("\nCopying results from device to host...");
	fflush(stdout);
	startTime(&timer);

	cudaMemcpy(G_h, G_d, 3 * sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));

	// Verify results
	printf("\nVerifying results...");
	fflush(stdout);
	verify(G_h, G_reference, num_bins);	

	// Free variables
	free(A_h); free(B_h); free(G_h);
	cudaFree(A_d); cudaFree(B_d); cudaFree(G_d);
	
    return 0;
}

#ifdef __cplusplus
}
#endif