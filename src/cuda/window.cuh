/*
 * All window-related operations.
 */

#ifndef IMAGE_H
#define IMAGE_H
#include "image.cuh"
#endif

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

// counts the number of pixels in the windoww
__device__ unsigned int count_window_pixels(unsigned int index, unsigned int height, unsigned int width, int window_radius);

// returns the min value across the rgb channels for all pixels in the window centered at (y, x)
__device__ float find_window_min(unsigned int index, pixel_t *image_pixels, unsigned int height, unsigned int width, int window_radius);

// computes the mean for the rgb color channels across the pixels in the window
__device__ void compute_window_mean(pixel_t *mean, unsigned int index, pixel_t *pixels, unsigned int height, unsigned int width, int window_radius);

// computes the variance for the rgb color channels across the pixels in the window
__device__ void compute_window_variance(pixel_t *variance, pixel_t *mean, unsigned int index, pixel_t *pixels, unsigned int height, unsigned int width, int window_radius);

// computes the dot product for a window across two sets of pixels
__device__ void compute_window_dot_product(pixel_t *dot_product, unsigned int index, pixel_t *pixels_X, pixel_t *pixels_Y, unsigned int height, unsigned int width, int window_radius);
