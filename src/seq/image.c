#include <stdio.h>
#include <stdlib.h>
#include "lib/lodepng.h"
#include "image.h"

// creates an image and converts from rgba bytes into pixels
image_t *init_image(unsigned int height, unsigned int width, unsigned char *bytes) {
    unsigned int num_pixels;
    unsigned int i;
    pixel_t pixel;

    num_pixels = height * width;

    image_t *image = malloc(sizeof(image_t));
    if (!image) {
        return NULL;
    }

    pixel_t *pixels = calloc(num_pixels, sizeof(pixel_t));
    if (!pixels) {
        free(image);
        return NULL;
    }

    image->height = height;
    image->width = width;

    for (i = 0; i < 4 * num_pixels; i += 4) {
        pixel.r = (float) bytes[i];
        pixel.g = (float) bytes[i + 1];
        pixel.b = (float) bytes[i + 2];
        pixel.a = (float) PIXEL_MAX_VALUE;

        pixels[i / 4] = pixel;
    }

    image->pixels = pixels;

    return image;
}

// frees an image
void free_image(image_t *image) {
    free(image->pixels);
    free(image);
}

// reads an image from file
image_t *read_image(char *filename) {
    unsigned int error;
    unsigned int height, width;
    unsigned char *bytes;
    
    error = lodepng_decode32_file(&bytes, &width, &height, filename);
    if (error) {
        fprintf(stderr, "Error %u: %s\n", error, lodepng_error_text(error));
        return NULL;
    }

    image_t *image = init_image(height, width, bytes);

    free(bytes);

    return image;
}

// writes the image to file
void write_image(image_t *image, char *filename) {
    unsigned int error;
    unsigned char *bytes;

    bytes = collapse_pixels(image);
    if (!bytes) {
        fprintf(stderr, "Error collapsing pixels\n");
        return;
    }

    error = lodepng_encode32_file(filename, bytes, image->width, image->height);
    if (error) {
        fprintf(stderr, "Error %u: %s\n", error, lodepng_error_text(error));
    }

    free(bytes);
}

// returns a copy of the image after replacing pixels
image_t *replace_pixels(image_t *image, pixel_t *pixels) {
    image_t *new_image;
    
    new_image = malloc(sizeof(image_t));
    if (!new_image) {
        return NULL;
    }

    new_image->height = image->height;
    new_image->width = image->width;
    new_image->pixels = pixels;

    return new_image;
}

// converts all pixels into a single array in rgba format
unsigned char *collapse_pixels(image_t *image) {
    unsigned int num_pixels, i, index;
    unsigned char *bytes;

    num_pixels = image->height * image->width;

    bytes = calloc(4 * num_pixels, sizeof(unsigned char));
    if (!bytes) {
        return NULL;
    }

    // index of next byte
    index = 0;
    for (i = 0; i < num_pixels; i++) {
        bytes[index] = (unsigned char) image->pixels[i].r;
        index++;
        bytes[index] = (unsigned char) image->pixels[i].g;
        index++;
        bytes[index] = (unsigned char) image->pixels[i].b;
        index++;
        bytes[index] = (unsigned char) image->pixels[i].a;
        index++;
    }

    return bytes;
}

// returns the mean of the pixel rgb values
float pixel_intensity(pixel_t *pixel) {
    return (pixel->r + pixel->g + pixel->b) / 3;
}

// finds the num_pixels brightest pixels in the given set and returns their indices
unsigned int *find_brightest_pixels(unsigned int num_pixels, pixel_t *pixels, unsigned int height, unsigned int width) {
    float min;
    unsigned int num_total_pixels;
    unsigned int len;
    unsigned int min_index;
    unsigned int i, j;
    unsigned int *indices;
    pixel_t pixel, temp_pixel;

    num_total_pixels = height * width;

    // indices of pixels in the haze opaque region
    indices = calloc(num_pixels, sizeof(unsigned int));
    if (!indices) {
        return NULL;
    }

    len = 0;
    min_index = 0;
    min = pixels[min_index].r;
    for (i = 0; i < num_total_pixels; i++) {
        pixel = pixels[i];

        // populate indices
        if (len < num_pixels) {
            indices[len] = i;

            if (pixel.r < min) {
                min_index = len;
                min = pixel.r;
            }

            len++;
        } else { // update indices
            if (pixel.r > min) {
                indices[min_index] = i;
                min = pixel.r;
                
                // update min
                for (j = 0; j < num_pixels; j++) {
                    temp_pixel = pixels[indices[j]];
                    if (temp_pixel.r < min) {
                        min_index = j;
                        min = temp_pixel.r;
                    }
                }
            }
        }
    }

    return indices;
}

// finds the brightest pixel in the image from the set of indices
unsigned int find_brightest_pixel(image_t *image, unsigned int *indices, unsigned int num_pixels) {
    float max;
    float temp;
    unsigned int i;
    unsigned int max_index;

    max_index = 0;
    max = pixel_intensity(&image->pixels[indices[max_index]]);
    for (i = 1; i < num_pixels; i++) {
        temp = pixel_intensity(&image->pixels[indices[i]]);
        if (temp > max) {
            max_index = i;
            max = temp;
        }
    }

    return indices[max_index];
}
