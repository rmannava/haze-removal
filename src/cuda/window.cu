#include "window.cuh"

// counts the number of pixels in the window
__device__ unsigned int count_window_pixels(unsigned int index, unsigned int height, unsigned int width, int window_radius) {
    int y, x;
    unsigned int y_min, y_max, x_min, x_max;

    y = index / width;
    x = index % width;

    y_min = MAX(0, y - window_radius);
    x_min = MAX(0, x - window_radius);
    y_max = MIN(height, y + window_radius + 1);
    x_max = MIN(width, x + window_radius + 1);

    return (y_max - y_min) * (x_max - x_min);
}

// returns the min value across the rgb channels for all pixels in the window
__device__ float find_window_min(unsigned int index, pixel_t *image_pixels, unsigned int height, unsigned int width, int window_radius) {
    float min;
    int y, x;
    unsigned int y_min, y_max, x_min, x_max;
    unsigned int i, j;
    pixel_t pixel;

    y = index / width;
    x = index % width;

    y_min = MAX(0, y - window_radius);
    x_min = MAX(0, x - window_radius);
    y_max = MIN(height, y + window_radius + 1);
    x_max = MIN(width, x + window_radius + 1);

    // min across window
    min = image_pixels[y_min * width + x_min].r;
    for (i = y_min; i < y_max; i++) {
        for (j = x_min; j < x_max; j++) {
            pixel = image_pixels[i * width + j];
            min = MIN(min, MIN(pixel.r, MIN(pixel.g, pixel.b)));
        }
    }

    return min;
    
}

// computes the mean for the rgb color channels across the pixels in the window
__device__ void compute_window_mean(pixel_t *mean, unsigned int index, pixel_t *pixels, unsigned int height, unsigned int width, int window_radius) {
    int y, x;
    unsigned int y_min, y_max, x_min, x_max;
    unsigned int i, j;
    unsigned int num_pixels;
    pixel_t pixel;

    y = index / width;
    x = index % width;

    y_min = MAX(0, y - window_radius);
    x_min = MAX(0, x - window_radius);
    y_max = MIN(height, y + window_radius + 1);
    x_max = MIN(width, x + window_radius + 1);

    mean->r = 0;
    mean->g = 0;
    mean->b = 0;
    for (i = y_min; i < y_max; i++) {
        for (j = x_min; j < x_max; j++) {
            pixel = pixels[i * width + j];

            mean->r += pixel.r;
            mean->g += pixel.g;
            mean->b += pixel.b;
            mean->a = PIXEL_MAX_VALUE;
        }
    }

    num_pixels = count_window_pixels(index, height, width, window_radius);

    mean->r /= num_pixels;
    mean->g /= num_pixels;
    mean->b /= num_pixels;
}

// computes the variance for the rgb color channels across the pixels in the window
__device__ void compute_window_variance(pixel_t *variance, pixel_t *mean, unsigned int index, pixel_t *pixels, unsigned int height, unsigned int width, int window_radius) {
    int y, x;
    unsigned int y_min, y_max, x_min, x_max;
    unsigned int i, j;
    unsigned int num_pixels;
    pixel_t pixel;

    y = index / width;
    x = index % width;

    y_min = MAX(0, y - window_radius);
    x_min = MAX(0, x - window_radius);
    y_max = MIN(height, y + window_radius + 1);
    x_max = MIN(width, x + window_radius + 1);

    variance->r = 0;
    variance->g = 0;
    variance->b = 0;
    for (i = y_min; i < y_max; i++) {
        for (j = x_min; j < x_max; j++) {
            pixel = pixels[i * width + j];

            variance->r += (pixel.r - mean->r) * (pixel.r - mean->r);
            variance->g += (pixel.g - mean->g) * (pixel.g - mean->g);
            variance->b += (pixel.b - mean->b) * (pixel.b - mean->b);
            variance->a = PIXEL_MAX_VALUE;
        }
    }

    num_pixels = count_window_pixels(index, height, width, window_radius);

    variance->r /= (num_pixels - 1);
    variance->g /= (num_pixels - 1);
    variance->b /= (num_pixels - 1);
}

// computes the dot product for a window across two sets of pixels
__device__ void compute_window_dot_product(pixel_t *dot_product, unsigned int index, pixel_t *pixels_X, pixel_t *pixels_Y, unsigned int height, unsigned int width, int window_radius) {
    int y, x;
    unsigned int y_min, y_max, x_min, x_max;
    unsigned int i, j;
    pixel_t pixel_x, pixel_y;

    y = index / width;
    x = index % width;

    y_min = MAX(0, y - window_radius);
    x_min = MAX(0, x - window_radius);
    y_max = MIN(height, y + window_radius + 1);
    x_max = MIN(width, x + window_radius + 1);

    dot_product->r = 0;
    dot_product->g = 0;
    dot_product->b = 0;
    for (i = y_min; i < y_max; i++) {
        for (j = x_min; j < x_max; j++) {
            pixel_x = pixels_X[i * width + j];
            pixel_y = pixels_Y[i * width + j];

            dot_product->r += pixel_x.r * pixel_y.r;
            dot_product->g += pixel_x.g * pixel_y.g;
            dot_product->b += pixel_x.b * pixel_y.b;
            dot_product->a = PIXEL_MAX_VALUE;
        }
    }
}
