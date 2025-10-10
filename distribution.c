#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void generate_random_coordinates(float *matrix, int t, int N) {
    for (int i = 0; i < t; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < 3; ++k) {
                matrix[i * N * 3 + j * 3 + k] = ((float)rand() / RAND_MAX) * 10.0f;
            }
        }
    }
}

void print_matrix(float *matrix, int t, int N) {
    for (int i = 0; i < t; ++i) {
        printf("Time step %d:\n", i);
        for (int j = 0; j < N; ++j) {
            printf("  [");
            for (int k = 0; k < 3; ++k) {
                printf(" %.4f", matrix[i * N * 3 + j * 3 + k]);
            }
            printf(" ]\n");
        }
    }
}

void radial_distribution_function(float *matrix, int t, int N) {
    float r_max = 10.0f; // Max distance to consider
    float bin_width = 0.1f; // Width of histogram bins
    int num_bins = (int)(r_max / bin_width);

    // Allocate and initialize histogram
    int *hist = calloc(num_bins, sizeof(int));

    // Histogram the distances
    for (int frame = 0; frame < t; ++frame) {
        for (int i = 0; i < N - 1; ++i) {
            float xi = matrix[frame * N * 3 + i * 3 + 0];
            float yi = matrix[frame * N * 3 + i * 3 + 1];
            float zi = matrix[frame * N * 3 + i * 3 + 2];

            for (int j = i + 1; j < N; ++j) {
                float xj = matrix[frame * N * 3 + j * 3 + 0];
                float yj = matrix[frame * N * 3 + j * 3 + 1];
                float zj = matrix[frame * N * 3 + j * 3 + 2];

                // Euclidean distance
                float dx = xi - xj;
                float dy = yi - yj;
                float dz = zi - zj;
                float r = sqrtf(dx*dx + dy*dy + dz*dz);

                if (r < r_max) {
                    int bin = (int)(r / bin_width);
                    hist[bin] += 2;  // Each pair counts for both particles
                }
            }
        }
    }

    // Normalize and print RDF
    float density = (float)(N * t) / (1000.0f);  // Will need to get volume externally
    float norm_factor = (float)(N * (N - 1) * t); // Normalization of data

    printf("\nRadial Distribution Function (g(r)):\n");
    for (int b = 0; b < num_bins; ++b) {
        float r_lower = b * bin_width;
        float r_upper = r_lower + bin_width;
        float r_mid = (r_lower + r_upper) / 2.0f;

        float shell_volume = (4.0f / 3.0f) * M_PI * (powf(r_upper, 3) - powf(r_lower, 3));
        float ideal_count = density * shell_volume;

        float g_r = (float)hist[b] / (norm_factor * ideal_count);
        printf("r = %.2f, g(r) = %.4f\n", r_mid, g_r);
    }

    free(hist);
}

int main() {
    int t = 5;   // Number of frames
    int N = 10;  // Number of atoms

    float *A_h = malloc(t * N * 3 * sizeof(float));

    srand((unsigned int)time(NULL));

    generate_random_coordinates(A_h, t, N);
    radial_distribution_function(A_h, t, N);

    free(A_h);
    return 0;
}
