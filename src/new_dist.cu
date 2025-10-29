#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

EXPORT int radial_distribution(
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

    // Zero the histogram
    for (int i = 0; i < num_bins; i++)
        g_r[i] = 0.0f;

    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {

            float dx = fabsf(coords_1[3*i + 0] - coords_2[3*j + 0]);
            float dy = fabsf(coords_1[3*i + 1] - coords_2[3*j + 1]);
            float dz = fabsf(coords_1[3*i + 2] - coords_2[3*j + 2]);

            // Apply minimum image convention
            if (dx > 0.5f * box[0]) dx = box[0] - dx;
            if (dy > 0.5f * box[1]) dy = box[1] - dy;
            if (dz > 0.5f * box[2]) dz = box[2] - dz;

            float r = sqrtf(dx*dx + dy*dy + dz*dz);
            int bin = (int)(r / bin_width);

            if (bin >= 0 && bin < num_bins)
                g_r[bin] += 1.0f;
        }
    }

    return 0;
}

#ifdef __cplusplus
}
#endif