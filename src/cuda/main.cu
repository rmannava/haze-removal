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
#include "image.cuh"
#endif

#include "window.cuh"

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

// check cuda function
#define CHECK_ERROR(error) check_error(error, __LINE__)

// check most recent cuda function
#define CHECK_LAST_ERROR() check_last_error(__LINE__)

// 1024 threads per block
#define THREAD_LIMIT 1024

// exits if there was an error
__host__ inline void check_error(cudaError_t error, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "\nCUDA Error at line %d: %s\n", line, cudaGetErrorString(error));
        exit(1);
    }
}

// exits if there was an error
__host__ inline void check_last_error(int line) {
    check_error(cudaGetLastError(), line);
}

// store transmission smoothing filter information
typedef struct {
    pixel_t a;
    pixel_t b;
} filter_elem_t;

// computes the dark channel of the image
__global__ void compute_dark_channel(pixel_t *dark_channel, pixel_t *image_pixels, unsigned int height, unsigned int width, int window_radius) {
    float min;
    unsigned int index;
    pixel_t pixel;

    index = blockIdx.x * blockDim.x + threadIdx.x;

    // check if this thread represents a valid pixel
    if (index >= height * width) {
        return;
    }

    min = find_window_min(index, image_pixels, height, width, window_radius);

    pixel.r = min;
    pixel.g = min;
    pixel.b = min;
    pixel.a = PIXEL_MAX_VALUE;

    dark_channel[index] = pixel;
}

// computes an estimate of the atmospheric light by finding the brightest pixel in the haze opaque region
__host__ void compute_atmospheric_light(pixel_t *atmos_light, image_t *image, pixel_t *dark_channel) {
    unsigned int num_pixels;
    unsigned int index;
    unsigned int *indices;

    // choose the size of the haze opaque region
    num_pixels = image->height * image->width * HAZE_OPAQUE_RATIO;

    // find pixels in the haze opaque region
    indices = find_brightest_pixels(num_pixels, dark_channel, image->height, image->width);
    if (!indices) {
        return;
    }

    // find the brightest pixel from the original image in the haze opaque region
    index = find_brightest_pixel(image, indices, num_pixels);
    *atmos_light = image->pixels[index];

    free(indices);
}

// computes the image normalized by the atmospheric light
__global__ void compute_norm(pixel_t *norm_pixels, pixel_t *image_pixels, unsigned int height, unsigned int width, pixel_t atmos_light) {
    unsigned int index;
    pixel_t image_pixel, transmission_pixel;

    index = blockIdx.x * blockDim.x + threadIdx.x;

    // check if this thread represents a valid pixel
    if (index >= height * width) {
        return;
    }

    // normalize pixel
    image_pixel = image_pixels[index];

    transmission_pixel.r = image_pixel.r / atmos_light.r;
    transmission_pixel.g = image_pixel.g / atmos_light.g;
    transmission_pixel.b = image_pixel.b / atmos_light.b;
    transmission_pixel.a = PIXEL_MAX_VALUE;

    norm_pixels[index] = transmission_pixel;
}

// computes the transmission in the image with the norm image dark channel
__global__ void compute_transmission(pixel_t *transmission_pixels, pixel_t *norm_pixels, unsigned int height, unsigned int width, int dark_channel_window_radius) {
    float min, transmission;
    unsigned int index;
    pixel_t transmission_pixel;

    index = blockIdx.x * blockDim.x + threadIdx.x;

    // check if this thread represents a valid pixel
    if (index >= height * width) {
        return;
    }

    // compute norm dark channel pixel value
    min = find_window_min(index, norm_pixels, height, width, dark_channel_window_radius);

    // compute transmission pixel value
    transmission = 1 - (1 - HAZE_RETENTION) * min;

    transmission_pixel.r = transmission;
    transmission_pixel.g = transmission;
    transmission_pixel.b = transmission;
    transmission_pixel.a = PIXEL_MAX_VALUE;

    transmission_pixels[index] = transmission_pixel;
}

// updates the filter elements in the window in place
__device__ void update_filter_window(filter_elem_t *filter, filter_elem_t *filter_elem, unsigned int index, unsigned int height, unsigned int width, int window_radius) {
    int y, x;
    unsigned int y_min, y_max, x_min, x_max;
    unsigned int i, j;
    unsigned int filter_index;

    y = index / width;
    x = index % width;

    y_min = MAX(0, y - window_radius);
    x_min = MAX(0, x - window_radius);
    y_max = MIN(height, y + window_radius + 1);
    x_max = MIN(width, x + window_radius + 1);

    // sum filter elements
    for (i = y_min; i < y_max; i++) {
        for (j = x_min; j < x_max; j++) {
            filter_index = i * width + j;

            atomicAdd(&filter[filter_index].a.r, filter_elem->a.r);
            atomicAdd(&filter[filter_index].a.g, filter_elem->a.g);
            atomicAdd(&filter[filter_index].a.b, filter_elem->a.b);
            filter[filter_index].a.a = PIXEL_MAX_VALUE;

            atomicAdd(&filter[filter_index].b.r, filter_elem->b.r);
            atomicAdd(&filter[filter_index].b.g, filter_elem->b.g);
            atomicAdd(&filter[filter_index].b.b, filter_elem->b.b);
            filter[filter_index].b.a = PIXEL_MAX_VALUE;
        }
    }
}

// computes the guided filter using the original image
__global__ void compute_filter(filter_elem_t *filter, pixel_t *image_pixels, pixel_t *transmission_pixels, unsigned int height, unsigned int width, int smooth_window_radius) {
    unsigned int index;
    unsigned int num_pixels;
    pixel_t image_mean, transmission_mean, image_variance, dot_product;
    filter_elem_t filter_elem;

    index = blockIdx.x * blockDim.x + threadIdx.x;

    // check if this thread represents a valid pixel
    if (index >= height * width) {
        return;
    }

    // compute window stats
    compute_window_mean(&image_mean, index, image_pixels, height, width, smooth_window_radius);
    compute_window_mean(&transmission_mean, index, transmission_pixels, height, width, smooth_window_radius);
    compute_window_variance(&image_variance, &image_mean, index, image_pixels, height, width, smooth_window_radius);
    compute_window_dot_product(&dot_product, index, image_pixels, transmission_pixels, height, width, smooth_window_radius);
    num_pixels = count_window_pixels(index, height, width, smooth_window_radius);

    // compute a
    filter_elem.a.r = ((dot_product.r / num_pixels) - (image_mean.r * transmission_mean.r)) / (image_variance.r + EDGE_VARIANCE);
    filter_elem.a.g = ((dot_product.g / num_pixels) - (image_mean.g * transmission_mean.g)) / (image_variance.g + EDGE_VARIANCE);
    filter_elem.a.b = ((dot_product.b / num_pixels) - (image_mean.b * transmission_mean.b)) / (image_variance.b + EDGE_VARIANCE);

    // compute b
    filter_elem.b.r = transmission_mean.r - filter_elem.a.r * image_mean.r;
    filter_elem.b.g = transmission_mean.g - filter_elem.a.g * image_mean.g;
    filter_elem.b.b = transmission_mean.b - filter_elem.a.b * image_mean.b;

    // compute filter coefficients
    update_filter_window(filter, &filter_elem, index, height, width, smooth_window_radius);
}

// computes the smooth transmission by applying the guided filter to the transmission
__global__ void compute_smooth_transmission(pixel_t *transmission_pixels, pixel_t *image_pixels, filter_elem_t *filter, unsigned int height, unsigned int width, int smooth_window_radius) {
    unsigned int index;
    unsigned int num_pixels;
    pixel_t image_pixel, transmission_pixel;
    filter_elem_t filter_elem;

    index = blockIdx.x * blockDim.x + threadIdx.x;

    // check if this thread represents a valid pixel
    if (index >= height * width) {
        return;
    }

    // compute smooth transmission pixel value
    image_pixel = image_pixels[index];
    filter_elem = filter[index];
    num_pixels = count_window_pixels(index, height, width, smooth_window_radius);

    transmission_pixel.r = (filter_elem.a.r / num_pixels) * image_pixel.r + (filter_elem.b.r / num_pixels);
    transmission_pixel.b = (filter_elem.a.b / num_pixels) * image_pixel.b + (filter_elem.b.b / num_pixels);
    transmission_pixel.g = (filter_elem.a.g / num_pixels) * image_pixel.g + (filter_elem.b.g / num_pixels);
    transmission_pixel.a = PIXEL_MAX_VALUE;

    transmission_pixels[index] = transmission_pixel;
}

// computes the scene radiance from the atmospheric light and the smooth transmission
__global__ void compute_scene_radiance(pixel_t *scene_radiance, pixel_t *image_pixels, unsigned int height, unsigned int width, pixel_t atmos_light, pixel_t *transmission_pixels) {
    unsigned int index;
    pixel_t pixel, image_pixel;

    index = blockIdx.x * blockDim.x + threadIdx.x;

    // check if this thread represents a valid pixel
    if (index >= height * width) {
        return;
    }

    // compute radiance
    image_pixel = image_pixels[index];

    // use abs to suppress artifacts - does a good job of hiding the problem if artifacts are rare
    pixel.r = fabs((image_pixel.r - atmos_light.r) / MAX(transmission_pixels[index].r, MIN_TRANSMISSION) + atmos_light.r);
    pixel.g = fabs((image_pixel.g - atmos_light.g) / MAX(transmission_pixels[index].g, MIN_TRANSMISSION) + atmos_light.g);
    pixel.b = fabs((image_pixel.b - atmos_light.b) / MAX(transmission_pixels[index].b, MIN_TRANSMISSION) + atmos_light.b);
    pixel.a = PIXEL_MAX_VALUE;

    scene_radiance[index] = pixel;
}

// returns a new image after haze removal
__host__ image_t *remove_haze(image_t *image) {
    unsigned int num_blocks;
    pixel_t atmos_light;
    pixel_t *dark_channel;
    pixel_t *scene_radiance;
    pixel_t *device_image_pixels;
    pixel_t *device_dark_channel;
    pixel_t *device_norm;
    pixel_t *device_transmission;
    pixel_t *device_scene_radiance;
    filter_elem_t *device_filter;
    image_t *dehazed_image;

    // divide by THREAD_LIMIT and round up
    num_blocks = (image->height * image->width + THREAD_LIMIT - 1) / THREAD_LIMIT;

    // initialize all required device memory
    CHECK_ERROR(cudaMalloc(&device_image_pixels, image->height * image->width * sizeof(pixel_t)));
    CHECK_ERROR(cudaMemcpy(device_image_pixels, (void *) image->pixels, image->height * image->width * sizeof(pixel_t), cudaMemcpyHostToDevice));

    CHECK_ERROR(cudaMalloc(&device_dark_channel, image->height * image->width * sizeof(pixel_t)));
    CHECK_ERROR(cudaMemset(device_dark_channel, 0, image->height * image->width * sizeof(pixel_t)));

    CHECK_ERROR(cudaMalloc(&device_norm, image->height * image->width * sizeof(pixel_t)));
    CHECK_ERROR(cudaMemset(device_norm, 0, image->height * image->width * sizeof(pixel_t)));

    CHECK_ERROR(cudaMalloc(&device_transmission, image->height * image->width * sizeof(pixel_t)));
    CHECK_ERROR(cudaMemset(device_transmission, 0, image->height * image->width * sizeof(pixel_t)));

    CHECK_ERROR(cudaMalloc(&device_scene_radiance, image->height * image->width * sizeof(pixel_t)));
    CHECK_ERROR(cudaMemset(device_scene_radiance, 0, image->height * image->width * sizeof(pixel_t)));

    CHECK_ERROR(cudaMalloc(&device_filter, image->height * image->width * sizeof(filter_elem_t)));
    CHECK_ERROR(cudaMemset(device_filter, 0, image->height * image->width * sizeof(filter_elem_t)));

    fprintf(stdout, "Computing dark channel...");

    // compute dark channel on device
    compute_dark_channel <<<num_blocks, THREAD_LIMIT, 0>>> (device_dark_channel, device_image_pixels, image->height, image->width, DARK_CHANNEL_WINDOW_RADIUS);

    // wait for all blocks
    cudaDeviceSynchronize();
    CHECK_LAST_ERROR();

    // copy dark channel to host
    dark_channel = (pixel_t *) calloc(image->height * image->width, sizeof(pixel_t));
    if (!dark_channel) {
        return NULL;
    }

    CHECK_ERROR(cudaMemcpy(dark_channel, device_dark_channel, image->height * image->width * sizeof(pixel_t), cudaMemcpyDeviceToHost));

    fprintf(stdout, "done\n");

    fprintf(stdout, "Computing atmospheric light...");

    // compute atmospheric light on host
    atmos_light.a = 0;
    compute_atmospheric_light(&atmos_light, image, dark_channel);
    if (!atmos_light.a) {
        fprintf(stderr, "\nError estimating atmospheric light\n");
        free(dark_channel);
        return NULL;
    }

    fprintf(stdout, "done\n");

    fprintf(stdout, "Computing transmission...");

    // compute normalized image on device
    compute_norm <<<num_blocks, THREAD_LIMIT, 0>>> (device_norm, device_image_pixels, image->height, image->width, atmos_light);

    // wait for all blocks
    cudaDeviceSynchronize();
    CHECK_LAST_ERROR();

    // compute transmission on device
    compute_transmission <<<num_blocks, THREAD_LIMIT, 0>>> (device_transmission, device_norm, image->height, image->width, DARK_CHANNEL_WINDOW_RADIUS);

    // wait for all blocks
    cudaDeviceSynchronize();
    CHECK_LAST_ERROR();

    fprintf(stdout, "done\n");

    fprintf(stdout, "Smoothing transmission...");

    // compute guided filter on device
    compute_filter <<<num_blocks, THREAD_LIMIT, 0>>> (device_filter, device_image_pixels, device_transmission, image->height, image->width, SMOOTH_WINDOW_RADIUS);

    // wait for all blocks
    cudaDeviceSynchronize();
    CHECK_LAST_ERROR();

    // compute smooth transmission on device
    compute_smooth_transmission <<<num_blocks, THREAD_LIMIT, 0>>> (device_transmission, device_image_pixels, device_filter, image->height, image->width, SMOOTH_WINDOW_RADIUS);

    // wait for all blocks
    cudaDeviceSynchronize();
    CHECK_LAST_ERROR();

    fprintf(stdout, "done\n");

    fprintf(stdout, "Computing scene radiance...");

    // compute scene radiance on device
    compute_scene_radiance <<<num_blocks, THREAD_LIMIT, 0>>> (device_scene_radiance, device_image_pixels, image->height, image->width, atmos_light, device_transmission);

    // wait for all blocks
    cudaDeviceSynchronize();
    CHECK_LAST_ERROR();

    // copy scene radiance to host
    scene_radiance = (pixel_t *) calloc(image->height * image->width, sizeof(pixel_t));
    if (!scene_radiance) {
        fprintf(stderr, "\nError computing scene radiance transmission\n");
        free(dark_channel);
        return NULL;
    }

    CHECK_ERROR(cudaMemcpy(scene_radiance, device_scene_radiance, image->height * image->width * sizeof(pixel_t), cudaMemcpyDeviceToHost));

    fprintf(stdout, "done\n");

    // construct dehazed image
    dehazed_image = replace_pixels(image, scene_radiance);

    free(dark_channel);

    CHECK_ERROR(cudaFree(device_image_pixels));
    CHECK_ERROR(cudaFree(device_dark_channel));
    CHECK_ERROR(cudaFree(device_norm));
    CHECK_ERROR(cudaFree(device_transmission));
    CHECK_ERROR(cudaFree(device_scene_radiance));
    CHECK_ERROR(cudaFree(device_filter));

    return dehazed_image;
}

// reads image, removes haze, writes resultant image
int main(int argc, char **argv) {
    char *input = NULL;
    char *output = NULL;
    const char *usage = "Usage: %s -i <input_filename> -o <output_filename>\n";
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

    fprintf(stdout, "CUDA Haze Removal on %s\n", input);

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
