/*
 * Performs single image haze removal using a dark channel prior and guided filter transmission smoothing.
 * Obsessively sets every alpha value to 255.
 *
 * Rama Mannava
 */

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <math.h>
#include <time.h>

#ifndef IMAGE_H
#define IMAGE_H
#include "image.h"
#endif

#include "window.h"

// higher window radius will improve dark channel accuracy but increase halo size
#define DARK_CHANNEL_WINDOW_RADIUS 5

// lower window radius will shrink halos but oversaturate colors
#define TRANSMISSION_WINDOW_RADIUS 5

// higher ratio will improve atmospheric light estimate but increase final haze level
#define HAZE_OPAQUE_RATIO 0.001

// will retain some haze to preserve depth information in dehazed image
#define HAZE_RETENTION 0.05

// higher minimum transmission will reduce noise and halo visibility but increase haze
#define MIN_TRANSMISSION 0.5

// higher window radius will suppress halos but introduce artifacts
#define SMOOTH_WINDOW_RADIUS 30

// lower threshold for pixel variance will increase smoothness of smooth transmission
#define EDGE_VARIANCE 0.001

// store transmission smoothing filter information
typedef struct {
    pixel_t a;
    pixel_t b;
} filter_elem_t;

// computes the pixels for the dark channel of the image
pixel_t *compute_dark_channel(image_t *image, int window_radius) {
    float min;
    unsigned int i, j;
    pixel_t pixel;
    pixel_t *dark_channel;

    dark_channel = calloc(image->height * image->width, sizeof(pixel_t));
    if (!dark_channel) {
        return NULL;
    }

    for (i = 0; i < image->height; i++) {
        for(j = 0; j < image->width; j++) {
            min = find_window_min(i, j, image, window_radius);

            pixel.r = min;
            pixel.g = min;
            pixel.b = min;
            pixel.a = PIXEL_MAX_VALUE;

            dark_channel[i * image->width + j] = pixel;
        }
    }

    return dark_channel;
}

// computes an estimate of atmospheric light by finding the brightest pixel in the haze opaque region
void compute_atmospheric_light(pixel_t *atmos_light, image_t *image, pixel_t *dark_channel) {
    unsigned int num_pixels;
    unsigned int index;
    unsigned int *indices;

    // choose the size of the haze opaque region
    num_pixels = image->height * image->width * HAZE_OPAQUE_RATIO;

    // find pixels in the haze opaque region
    indices = find_brightest_pixels(num_pixels, image->height, image->width, dark_channel);
    if (!indices) {
        return;
    }

    // find the brightest pixel from the original image in the haze opaque region
    index = find_brightest_pixel(indices, num_pixels, image);
    *atmos_light = image->pixels[index];

    free(indices);
}

// computes the pixels for the dark channel of the image normalized for atmospheric light
pixel_t *compute_norm_dark_channel(image_t *image, pixel_t *atmos_light, int window_radius) {
    unsigned int i;
    pixel_t pixel;
    pixel_t *norm_pixels;
    pixel_t *norm_dark_channel;
    image_t *norm_image;

    norm_pixels = calloc(image->height * image->width, sizeof(pixel_t));
    if (!norm_pixels) {
        return NULL;
    }

    // normalize all pixels
    for (i = 0; i < image->height * image->width; i++) {
        pixel = image->pixels[i];

        norm_pixels[i].r = pixel.r / atmos_light->r;
        norm_pixels[i].g = pixel.g / atmos_light->g;
        norm_pixels[i].b = pixel.b / atmos_light->b;
        norm_pixels[i].a = PIXEL_MAX_VALUE;
    }
    norm_image = replace_pixels(image, norm_pixels);

    norm_dark_channel = compute_dark_channel(norm_image, window_radius);

    free_image(norm_image);

    return norm_dark_channel;
}

// computes the transmission in the image with respect to the atmospheric light
pixel_t *compute_transmission(image_t *image, pixel_t *atmos_light) {
    float temp;
    unsigned int i;
    pixel_t *norm_dark_channel;
    pixel_t *transmission;

    norm_dark_channel = compute_norm_dark_channel(image, atmos_light, TRANSMISSION_WINDOW_RADIUS);

    transmission = calloc(image->height * image->width, sizeof(pixel_t));
    if (!transmission) {
        free(norm_dark_channel);
        return NULL;
    }

    // compute transmission for each pixel
    for (i = 0; i < image->height * image->width; i++) {
        temp = 1 - (1 - HAZE_RETENTION) * norm_dark_channel[i].r;

        transmission[i].r = temp;
        transmission[i].g = temp;
        transmission[i].b = temp;
        transmission[i].a = PIXEL_MAX_VALUE;
    }

    free(norm_dark_channel);

    return transmission;
}

// updates filter elements in the window in place
void update_filter_window(int y, int x, filter_elem_t *filter, filter_elem_t *filter_elem, unsigned int height, unsigned int width, int window_radius) {
    unsigned int y_min, y_max, x_min, x_max;
    unsigned int i, j;
    filter_elem_t temp;

    y_min = MAX(0, y - window_radius);
    x_min = MAX(0, x - window_radius);
    y_max = MIN(height, y + window_radius + 1);
    x_max = MIN(width, x + window_radius + 1);

    for (i = y_min; i < y_max; i++) {
        for (j = x_min; j < x_max; j++) {
            temp = filter[i * width + j];

            temp.a.r += filter_elem->a.r;
            temp.a.g += filter_elem->a.g;
            temp.a.b += filter_elem->a.b;
            temp.a.a = PIXEL_MAX_VALUE;

            temp.b.r += filter_elem->b.r;
            temp.b.g += filter_elem->b.g;
            temp.b.b += filter_elem->b.b;
            temp.b.a = PIXEL_MAX_VALUE;

            filter[i * width + j] = temp;
        }
    }
}

// computes a smooth transmission by reducing blockiness and introducing edges from original image
pixel_t *compute_smooth_transmission(image_t *image, pixel_t *transmission, int window_size) {
    unsigned int i, j;
    unsigned int num_pixels;
    pixel_t image_mean, transmission_mean, image_variance, dot_product;
    pixel_t pixel, transmission_pixel;
    pixel_t *smooth_transmission;
    filter_elem_t filter_elem;
    filter_elem_t *filter;

    filter = calloc(image->height * image->width, sizeof(filter_elem_t));
    if (!filter) {
        return NULL;
    }

    smooth_transmission = calloc(image->height * image->width, sizeof(pixel_t));
    if (!smooth_transmission) {
        free(filter);
        return NULL;
    }

    // compute filter coefficients for all pixels in each window
    for (i = 0; i < image->height; i++) {
        for (j = 0; j < image->width; j++) {
            compute_window_mean(&image_mean, i, j, image->pixels, image->height, image->width, window_size);
            compute_window_mean(&transmission_mean, i, j, transmission, image->height, image->width, window_size);
            compute_window_variance(&image_mean, &image_variance, i, j, image->pixels, image->height, image->width, window_size);
            compute_window_dot_product(&dot_product, i, j, image->pixels, transmission, image->height, image->width, window_size);
            num_pixels = count_window_pixels(i, j, image->height, image->width, window_size);

            // compute a
            filter_elem.a.r = ((dot_product.r / num_pixels) - (image_mean.r * transmission_mean.r)) / (image_variance.r + EDGE_VARIANCE);
            filter_elem.a.g = ((dot_product.g / num_pixels) - (image_mean.g * transmission_mean.g)) / (image_variance.g + EDGE_VARIANCE);
            filter_elem.a.b = ((dot_product.b / num_pixels) - (image_mean.b * transmission_mean.b)) / (image_variance.b + EDGE_VARIANCE);

            // compute b
            filter_elem.b.r = transmission_mean.r - filter_elem.a.r * image_mean.r;
            filter_elem.b.g = transmission_mean.g - filter_elem.a.g * image_mean.g;
            filter_elem.b.b = transmission_mean.b - filter_elem.a.b * image_mean.b;

            update_filter_window(i, j, filter, &filter_elem, image->height, image->width, window_size);
        }
    }

    // calculate smooth transmission pixels from filter
    for (i = 0; i < image->height; i++) {
        for (j = 0; j < image->width; j++) {
            pixel = image->pixels[i * image->width + j];
            transmission_pixel = smooth_transmission[i * image->width + j];
            filter_elem = filter[i * image->width + j];
            num_pixels = count_window_pixels(i, j, image->height, image->width, window_size);

            transmission_pixel.r = (filter_elem.a.r / num_pixels) * pixel.r + (filter_elem.b.r / num_pixels);
            transmission_pixel.b = (filter_elem.a.b / num_pixels) * pixel.b + (filter_elem.b.b / num_pixels);
            transmission_pixel.g = (filter_elem.a.g / num_pixels) * pixel.g + (filter_elem.b.g / num_pixels);
            transmission_pixel.a = PIXEL_MAX_VALUE;

            smooth_transmission[i * image->width + j] = transmission_pixel;
        }
    }

    free(filter);

    return smooth_transmission;
}

// computes scene radiance from original hazy image
pixel_t *compute_scene_radiance(image_t *image, pixel_t *atmos_light, pixel_t *transmission) {
    unsigned int i;
    pixel_t pixel;
    pixel_t *scene_radiance;

    scene_radiance = calloc(image->height * image->width, sizeof(pixel_t));
    if (!scene_radiance) {
        return NULL;
    }

    // compute radiance at each pixel
    for (i = 0; i < image->height * image->width; i++) {
        pixel = image->pixels[i];

        // use abs to suppress artifacts - does a good job of hiding the problem if artifacts are rare
        scene_radiance[i].r = fabs((pixel.r - atmos_light->r) / MAX(transmission[i].r, MIN_TRANSMISSION) + atmos_light->r);
        scene_radiance[i].g = fabs((pixel.g - atmos_light->g) / MAX(transmission[i].g, MIN_TRANSMISSION) + atmos_light->g);
        scene_radiance[i].b = fabs((pixel.b - atmos_light->b) / MAX(transmission[i].b, MIN_TRANSMISSION) + atmos_light->b);
        scene_radiance[i].a = PIXEL_MAX_VALUE;
    }

    return scene_radiance;
}

// returns a new image after haze removal
image_t *remove_haze(image_t *image) {
    pixel_t atmos_light;
    pixel_t *dark_channel;
    pixel_t *transmission;
    pixel_t *smooth_transmission;
    pixel_t *scene_radiance;
    image_t *dehazed_image;

    fprintf(stdout, "Computing dark channel...");
    dark_channel = compute_dark_channel(image, DARK_CHANNEL_WINDOW_RADIUS);
    fprintf(stdout, "done\n");
    if (!dark_channel) {
        fprintf(stderr, "Error computing dark channel\n");
        return NULL;
    }

    atmos_light.a = 0;
    fprintf(stdout, "Computing atmospheric light...");
    compute_atmospheric_light(&atmos_light, image, dark_channel);
    fprintf(stdout, "done\n");
    if (!atmos_light.a) {
        fprintf(stderr, "Error estimating atmospheric light\n");
        free(dark_channel);
        return NULL;
    }

    fprintf(stdout, "Computing transmission...");
    transmission = compute_transmission(image, &atmos_light);
    fprintf(stdout, "done\n");
    if (!transmission) {
        fprintf(stderr, "Error computing transmission\n");
        free(dark_channel);
        return NULL;
    }

    fprintf(stdout, "Smoothing transmission...");
    smooth_transmission = compute_smooth_transmission(image, transmission, SMOOTH_WINDOW_RADIUS);
    fprintf(stdout, "done\n");
    if (!smooth_transmission) {
        fprintf(stderr, "Error computing smooth transmission\n");
        free(dark_channel);
        free(transmission);
        return NULL;
    }

    fprintf(stdout, "Computing scene radiance...");
    scene_radiance = compute_scene_radiance(image, &atmos_light, smooth_transmission);
    fprintf(stdout, "done\n");

    dehazed_image = replace_pixels(image, scene_radiance);

    free(dark_channel);
    free(transmission);
    free(smooth_transmission);

    return dehazed_image;
}

// reads image, removes haze, writes resultant image
int main(int argc, char **argv) {
    char *input = NULL;
    char *output = NULL;
    char *usage = "Usage: %s -i <input_filename> -o <output_filename>\n";
    int opt;
    clock_t start_time, end_time;
    image_t *image;
    image_t *new_image;

    // disable stdout buffering
    setbuf(stdout, NULL);

    // parse arguments
    while ((opt = getopt(argc, argv, "hi:o:")) != -1) {
        switch(opt) {
            case 'h':
                fprintf(stdout, usage, argv[0]);
                return 0;
            case 'i':
                input = optarg;
                break;
            case 'o':
                output = optarg;
                break;
            case '?':
                if (optopt == 'i') {
                    fprintf(stderr, "Option -i requires an input image filename.\n");
                } else if (optopt == 'o') {
                    fprintf(stderr, "Option -o requires an output image filename.\n");
                } else {
                    fprintf(stdout, usage, argv[0]);
                }
                return 1;
        }
    }

    // check for required arguments
    if (!input || !output) {
        fprintf(stderr, "Haze Removal requires an input image and an output image.\n");
        fprintf(stdout, usage, argv[0]);
        return 1;
    }

    fprintf(stdout, "Sequential Haze Removal on %s\n", input);

    image = read_image(input);
    if (!image) {
        return 1;
    }

    fprintf(stdout, "Parsed %ux%u image\n\n", image->width, image->height);

    start_time = clock();
    new_image = remove_haze(image);
    end_time = clock();
    if (!new_image) {
        free_image(image);
        return 1;
    }

    fprintf(stdout, "\nWriting Image to %s\n", output); 

    write_image(new_image, output);

    fprintf(stdout, "\nComputation Time: %.3f seconds\n", ((float) end_time - start_time) / CLOCKS_PER_SEC);

    free_image(image);
    free_image(new_image);

    return 0;
}
