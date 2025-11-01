/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "support.h"

void startTime(Timer* timer) {
  timer->startTime = std::chrono::high_resolution_clock::now();
}


void stopTime(Timer* timer) {
  timer->endTime = std::chrono::high_resolution_clock::now();
}

float elapsedTime(Timer timer) {
  return std::chrono::duration<float>(timer.endTime - timer.startTime).count();
}

// Not under Copyright

void generateCoordinates(float *coords, int n, float box_size) {
    for (int i = 0; i < n; i++) {
        coords[3 * i + 0] = ((float)rand() / RAND_MAX) * box_size;
        coords[3 * i + 1] = ((float)rand() / RAND_MAX) * box_size;
        coords[3 * i + 2] = ((float)rand() / RAND_MAX) * box_size;
    }
}