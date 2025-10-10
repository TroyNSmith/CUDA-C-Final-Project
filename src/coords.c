#include <stdio.h>
#include <stdlib.h>
#include "xdrfile_xtc.h"

#include <stdio.h>
#include <stdlib.h>
#include "xdrfile_xtc.h"
#include "xdrfile.h"

int read_xtc_file(const char *xtc_file) {
    int natoms;
    // Fetch the number of atoms
    if (read_xtc_natoms(xtc_file, &natoms) != 0 || natoms <= 0) {
        fprintf(stderr, "Failed to get number of atoms (%d) from %s\n", natoms, xtc_file);
        return -1;
    }

    XDRFILE *xd = xdrfile_open(xtc_file, "r");
    if (!xd) {
        fprintf(stderr, " [ Error ] Failed to open file: %s\n", xtc_file);
        return -1;
    }

    rvec *x = malloc(sizeof(rvec) * natoms);
    if (!x) {
        fprintf(stderr, "[ Error ] Failed to allocate memory.\n");
        xdrfile_close(xd);
        return -1;
    }

    matrix box;
    int step;
    float time;
    float prec;
    int frame_count = 0;

    if (read_xtc(xd, natoms, &step, &time, box, x, &prec) == exdrOK) {
        printf("Frame %d — Step: %d, Time: %f\n", frame_count++, step, time);
        printf("Atom 1: %f %f %f\n", x[0][0], x[0][1], x[0][2]);
    } else {
        fprintf(stderr, " [ Error ] Failed to read first frame.\n");
    }

    free(x);
    xdrfile_close(xd);
    return natoms;
}