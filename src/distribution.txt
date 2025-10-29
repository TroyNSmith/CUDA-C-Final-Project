#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "xdrfile.h"
#include "xdrfile_xtc.h"

extern "C" {

#ifndef EXPORT
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif
#endif

// To-do: Dynamically allocate matrix to fit number of residues
#define MAX_RES 2000
#define PI 3.14159265

/*
Takes in an xtc files, an array of pairs, the length of pairs, bins, g_r, and the number of bins. The xtc files gives the number atoms in the simulation,
and is put into an accessible file descriptor to be read into values in a loop that takes in the file to program.
*/
EXPORT int radial_distribution(const char *xtc_file, float *pairs, int pairs_len, float *bins, float *g_r, int num_bins) {
    int natoms;
    if (read_xtc_natoms((char *)xtc_file, &natoms) != 0 || natoms <= 0) {
        fprintf(stderr, "[ Error] Failed to read number of atoms from %s\n", xtc_file);
        return -1;
    }

    rvec *x = (rvec*)malloc(sizeof(rvec) * natoms);
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

    int *hist = (int*)calloc(num_bins, sizeof(int));
    if (!hist) {
        fprintf(stderr, "[ Error] Histogram memory allocation failed\n");
        free(x);
        xdrfile_close(xdr);
        return -1;
    }

    // Number of residues encountered
    int N = 0;

    // While loop to read in values from the xtc file
    /*
    Best guess on the values:
    xdr: file descriptor for the xtc file (used to read data, not in loop)
    natoms: number of atoms in the simulation (Just used for skipping invalid entries)
    step: current step in the simulation (unused)
    time: current time in the simulation (unused)
    box: simulation box dimensions (Id guess its the same for each step, unused in loop but used after loop)
    x: array of atom coordinates (Really only useful value)
    prec: precision of the coordinates (unsed)
    */
    while (read_xtc(xdr, natoms, &step, &time, box, x, &prec) == exdrOK) {
        //First two lines fill com(center of mass) and total mass with 0s. 
        //Com is a array of 3D coordinates for each residue, total mass is an array of the actual mass of the reside
        memset(com, 0, sizeof(com));
        memset(total_mass, 0, sizeof(total_mass));

        // Loop over every atom in pairs
        for (int i = 0; i < pairs_len; i += 3) {
            // Pairs is an array of tuples (atom_id, res_id, mass)
            int atom_id = (int)pairs[i];
            int res_id = (int)pairs[i + 1];
            float mass = pairs[i + 2];

            // Skip invalid entries
            if (atom_id < 0 || atom_id >= natoms || res_id < 0 || res_id >= MAX_RES) {
                continue;
            }

            // Calculate center of mass for each residue. My best guess is that x is an array of coordinates for each atom
            // So the multiplication here is just mass-weighting along each basis vector
            com[res_id][0] += x[atom_id][0] * mass;
            com[res_id][1] += x[atom_id][1] * mass;
            com[res_id][2] += x[atom_id][2] * mass;
            total_mass[res_id] += mass;

            // I guess we have the assumption the res_id's will increase, and we want to keep track of them.
            // However multiple atoms in a residue are encountered, so only update N(Number of residues) if it increases
            if (res_id + 1 > N) N = res_id + 1;
        }

        // Loop over every residue to get center of mass (mass on each axis/ total mass)
        for (int r = 0; r < N; ++r) {
            if (total_mass[r] > 0.0f) {
                com[r][0] /= total_mass[r];
                com[r][1] /= total_mass[r];
                com[r][2] /= total_mass[r];
            }
        }

        // This is the actual residue distance calculation loop
        for (int i = 0; i < N; ++i) {
            // Skip empty residues
            if (total_mass[i] == 0.0f) continue;
            // Double loop since its a pairwise calculation
            for (int j = i + 1; j < N; ++j) {
                // Skip empty residues
                if (total_mass[j] == 0.0f) continue;
                // dx = com[i][0] - com[j][0] > a / 2, then dx = a - | dx - dy |, where a is the length of the box in x-axis
                // What is above?

                // Difference along each axis
                float dx = com[i][0] - com[j][0];
                float dy = com[i][1] - com[j][1];
                float dz = com[i][2] - com[j][2];
                // Euclidean distance
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);

                // If distance is less than a tenth of the number of bins, put it in the histogram?
                // Why this condition? I see we multiply it by 10, but I'm not sure why that is relevant either
                // Maybe it normalizes against distances that are too large?
                if (dist < num_bins * 0.1f) {
                    int bin = (int)(dist / 0.1f);
                    hist[bin] += 2;
                }
            }
        }

        // Track number of frames processed
        frame_count++;
    }

    // Volume is length * width * height lol
    float volume = box[0][0] * box[1][1] * box[2][2];
    // Density is number of residues / volume
    float density = (float)(N) / volume;
    // Normalization factor for g(r), Seems to be number of unique pairs of residues * number of frames, unsure of the why of this
    float norm_factor = (float)(N * (N - 1) / 2 * frame_count);

    // Loop to calculate g(r), looping over each bins in the histogram
    for (int b = 0; b < num_bins; ++b) {
        // To-Do: Add bin widths

        // I guess each bin is 0.01 width
        float r_lower = b * 0.01f;
        float r_upper = r_lower + 0.01f;
        // Midpoint of the bin
        float r_mid = (r_lower + r_upper) / 2.0f;

        // Volume of the spherical shell for this bin
        // Imagine an onion, each layer is one of the bins we worked on.
        float shell_volume = (4.0f / 3.0f) * PI * (powf(r_upper, 3) - powf(r_lower, 3));
        // Ideal count of pairs in this shell based on density
        // We find the expected number of residues in each layer of the onion
        float ideal_count = density * shell_volume;

        bins[b] = r_mid; // I guess bins is an array of bin centers
        g_r[b] = (float)hist[b] / (norm_factor * ideal_count + 1e-6f); // g_r is the array of radial distribution values
        // Then look at the ratio of the actual number of residues in that layer of the onion to this normalized ideal count
    }

    free(hist);
    free(x);
    xdrfile_close(xdr);
    return 0;
}
}
