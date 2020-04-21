#include "window.h"

// counts the number of pixels in the windoww
unsigned int count_window_pixels(int y, int x, unsigned int height, unsigned int width, int window_radius) {
    unsigned int y_min, y_max, x_min, x_max;

    y_min = MAX(0, y - window_radius);
    x_min = MAX(0, x - window_radius);
    y_max = MIN(height, y + window_radius + 1);
    x_max = MIN(width, x + window_radius + 1);

    return (y_max - y_min) * (x_max - x_min);
}

// returns the min value across the rgb channels for all pixels in the window centered at (y, x)
float find_window_min(int y, int x, image_t *image, int window_radius) {
    float min;
    unsigned int y_min, y_max, x_min, x_max;
    unsigned int i, j;
    pixel_t pixel;

    y_min = MAX(0, y - window_radius);
    x_min = MAX(0, x - window_radius);
    y_max = MIN(image->height, y + window_radius + 1);
    x_max = MIN(image->width, x + window_radius + 1);

    // min across window
    min = image->pixels[y_min * image->width + x_min].r;
    for (i = y_min; i < y_max; i++) {
        for (j = x_min; j < x_max; j++) {
            pixel = image->pixels[i * image->width + j];
            min = MIN(min, MIN(pixel.r, MIN(pixel.g, pixel.b)));
        }
    }

    return min;
    
}

// computes the mean for the rgb color channels across the pixels in the window
void compute_window_mean(pixel_t *mean, int y, int x, pixel_t *pixels, unsigned int height, unsigned int width, int window_radius) {
    unsigned int y_min, y_max, x_min, x_max;
    unsigned int i, j;
    unsigned int num_pixels;
    pixel_t pixel;

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

    num_pixels = count_window_pixels(y, x, height, width, window_radius);

    mean->r /= num_pixels;
    mean->g /= num_pixels;
    mean->b /= num_pixels;
}

// computes the variance for the rgb color channels across the pixels in the window
void compute_window_variance(pixel_t *mean, pixel_t *variance, int y, int x, pixel_t *pixels, unsigned int height, unsigned int width, int window_radius) {
    unsigned int y_min, y_max, x_min, x_max;
    unsigned int i, j;
    unsigned int num_pixels;
    pixel_t pixel;

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

    num_pixels = count_window_pixels(y, x, height, width, window_radius);

    variance->r /= (num_pixels - 1);
    variance->g /= (num_pixels - 1);
    variance->b /= (num_pixels - 1);
}

// computes the dot product for a window across two sets of pixels
void compute_window_dot_product(pixel_t *dot_product, int y, int x, pixel_t *pixels_X, pixel_t *pixels_Y, unsigned int height, unsigned int width, int window_radius) {
    unsigned int y_min, y_max, x_min, x_max;
    unsigned int i, j;
    pixel_t pixel_x, pixel_y;

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
