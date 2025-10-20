#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "xdrfile.h"
#include "xdrfile_xtc.h"

// To-do: Dynamically allocate matrix to fit number of residues
#define MAX_RES 2000
#define PI 3.14159265

void radial_distribution(const char *xtc_file, float *pairs, int pairs_len, float *bins, float *g_r, int num_bins) {
    int natoms;
    if (read_xtc_natoms((char *)xtc_file, &natoms) != 0 || natoms <= 0) {
        fprintf(stderr, "[ Error] Failed to read number of atoms from %s\n", xtc_file);
        return -1;
    }

    rvec *x = malloc(sizeof(rvec) * natoms);
    if (!x) {
        fprintf(stderr, "[ Error] Failed to allocate memory.\n");
        return -1;
    }

    XDRFILE *xdr = xdrfile_open(xtc_file, "r");
    if (!xdr) {
        fprintf(stderr, "[ Error ] Failed to open file: %s\n", xtc_file);
        free(x);
        return -1;
    }

    matrix box;
    int step;
    float time;
    float prec;
    int frame_count = 0;

    float com[MAX_RES][3] = {0};
    float total_mass[MAX_RES] = {0};

    int *hist = calloc(num_bins, sizeof(int));
    if (!hist) {
        fprintf(stderr, "[ Error] Histogram memory allocation failed\n");
        free(x);
        xdrfile_close(xdr);
        return -1;
    }

    int N = 0;

    while (read_xtc(xdr, natoms, &step, &time, box, x, &prec) == exdrOK) {
        memset(com, 0, sizeof(com));
        memset(total_mass, 0, sizeof(total_mass));

        for (int i = 0; i < pairs_len; i += 3) {
            int atom_id = (int)pairs[i];
            int res_id = (int)pairs[i + 1];
            float mass = pairs[i + 2];

            if (atom_id < 0 || atom_id >= natoms || res_id < 0 || res_id >= MAX_RES) {
                continue;
            }

            com[res_id][0] += x[atom_id][0] * mass;
            com[res_id][1] += x[atom_id][1] * mass;
            com[res_id][2] += x[atom_id][2] * mass;
            total_mass[res_id] += mass;

            if (res_id + 1 > N) N = res_id + 1;
        }

        for (int r = 0; r < N; ++r) {
            if (total_mass[r] > 0.0f) {
                com[r][0] /= total_mass[r];
                com[r][1] /= total_mass[r];
                com[r][2] /= total_mass[r];
            }
        }

        for (int i = 0; i < N; ++i) {
            if (total_mass[i] == 0.0f) continue;

            for (int j = i + 1; j < N; ++j) {
                if (total_mass[j] == 0.0f) continue;
                // dx = com[i][0] - com[j][0] > a / 2, then dx = a - | dx - dy |, where a is the length of the box in x-axis

                float dx = com[i][0] - com[j][0];
                float dy = com[i][1] - com[j][1];
                float dz = com[i][2] - com[j][2];
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);

                if (dist < num_bins * 0.1f) {
                    int bin = (int)(dist / 0.1f);
                    hist[bin] += 2;
                }
            }
        }

        frame_count++;
    }

    float volume = box[0][0] * box[1][1] * box[2][2];
    float density = (float)(N) / volume;
    float norm_factor = (float)(N * (N - 1) / 2 * frame_count);

    for (int b = 0; b < num_bins; ++b) {
        // To-Do: Add bin widths
        float r_lower = b * 0.01f;
        float r_upper = r_lower + 0.01f;
        float r_mid = (r_lower + r_upper) / 2.0f;

        float shell_volume = (4.0f / 3.0f) * PI * (powf(r_upper, 3) - powf(r_lower, 3));
        float ideal_count = density * shell_volume;

        bins[b] = r_mid;
        g_r[b] = (float)hist[b] / (norm_factor * ideal_count + 1e-6f);
    }

    free(hist);
    free(x);
    xdrfile_close(xdr);
}
