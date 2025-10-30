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
  gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
  gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
  return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

// Not under Copyright

void generateCoordinates(float *coords, int n, float box_size) {
    for (int i = 0; i < n; i++) {
        coords[3 * i + 0] = ((float)rand() / RAND_MAX) * box_size;
        coords[3 * i + 1] = ((float)rand() / RAND_MAX) * box_size;
        coords[3 * i + 2] = ((float)rand() / RAND_MAX) * box_size;
    }
}