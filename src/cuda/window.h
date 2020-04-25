/*
 * All window-related operations.
 */

#ifndef IMAGE_H
#define IMAGE_H
#include "image.h"
#endif

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

// counts the number of pixels in the windoww
unsigned int count_window_pixels(int y, int x, unsigned int height, unsigned int width, int window_radius);

// returns the min value across the rgb channels for all pixels in the window centered at (y, x)
float find_window_min(int y, int x, image_t *image, int window_radius);

// computes the mean for the rgb color channels across the pixels in the window
void compute_window_mean(pixel_t *mean, int y, int x, pixel_t *pixels, unsigned int height, unsigned int width, int window_radius);

// computes the variance for the rgb color channels across the pixels in the window
void compute_window_variance(pixel_t *mean, pixel_t *variance, int y, int x, pixel_t *pixels, unsigned int height, unsigned int width, int window_radius);

// computes the dot product for a window across two sets of pixels
void compute_window_dot_product(pixel_t *dot_product, int y, int x, pixel_t *pixels_X, pixel_t *pixels_Y, unsigned int height, unsigned int width, int window_radius);
