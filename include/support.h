/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#ifndef __FILEH__
#define __FILEH__

#include <chrono>

typedef struct {
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point endTime;
} Timer;

#ifdef __cplusplus
extern "C" {
#endif

void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);
void generateCoordinates(float *coords, int n, float box_size);

#ifdef __cplusplus
}
#endif

#define FATAL(msg, ...)                                                       \
    do {                                                                      \
        printf("[%s:%d\n%s]", __FILE__, __LINE__, msg);                       \
        exit(-1);                                                             \
    } while(0)

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

#endif
