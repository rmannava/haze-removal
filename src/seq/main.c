#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include "../lib/lodepng.h"

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

// patch size for dark channel and transmission
// higher patch size will improve dark channel accuracy but introduce depth artifacts
#define DARK_CHANNEL_PATCH_SIZE 15
// lower patch size will tighten constant transmission assumption but oversaturate colors
#define TRANSMISSION_PATCH_SIZE 15

// fraction of image size for haze-opaque region
#define HAZE_OPAQUE_SIZE 0.001

// will retain 5% of haze to preserve depth information in dehazed image
#define HAZE_RETENTION 0.05

// minimum transmission to reduce noise visibility
#define MIN_TRANSMISSION 0.1

// window size for transmission smoothing
#define WINDOW_SIZE 3

// lower threshold for pixel variance will increase edges in smooth transmission
#define EDGE_VARIANCE 1

// store rgba as floats for precise math
typedef struct {
    float r;
    float g;
    float b;
    float a;
} pixel_t;

// store image information
typedef struct {
    unsigned int height;
    unsigned int width;
    pixel_t *pixels;
} image_t;

// creates an image and converts from rgba bytes into pixels
image_t *init_image(unsigned int height, unsigned int width, unsigned char *bytes) {
    unsigned int i;
    pixel_t pixel;

    image_t *image = malloc(sizeof(image_t));
    if (!image) {
        return NULL;
    }

    image->height = height;
    image->width = width;

    pixel_t *pixels = calloc(height * width, sizeof(pixel_t));
    if (!pixels) {
        free(image);
        return NULL;
    }

    for (i = 0; i < 4 * height * width; i += 4) {
        pixel.r = (float) bytes[i];
        pixel.g = (float) bytes[i + 1];
        pixel.b = (float) bytes[i + 2];
        pixel.a = (float) bytes[i + 3];

        pixels[i / 4] = pixel;
    }

    image->pixels = pixels;

    return image;
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

// converts all pixels into a single array in rgba format
unsigned char *collapse_pixels(image_t *image) {
    unsigned int i, index;
    unsigned char *bytes;

    bytes = calloc(4 * image->height * image->width, sizeof(unsigned char));
    if (!bytes) {
        return NULL;
    }

    // index of next byte
    index = 0;
    for (i = 0; i < image->height * image->width; i++) {
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

// returns the min value across the rgb channels for all pixels in the patch centered at (y, x)
float find_patch_min(int y, int x, image_t *image, int patch_size) {
    float min;
    unsigned int y_min, y_max, x_min, x_max;
    unsigned int i, j;
    pixel_t pixel;

    y_min = MAX(0, y - patch_size / 2);
    x_min = MAX(0, x - patch_size / 2);
    y_max = MIN(image->height, y + patch_size / 2 + 1);
    x_max = MIN(image->width, x + patch_size / 2 + 1);

    // min across patch
    min = image->pixels[y_min * image->width + x_min].r;
    for (i = y_min; i < y_max; i++) {
        for (j = x_min; j < x_max; j++) {
            pixel = image->pixels[i * image->width + j];
            min = MIN(min, MIN(pixel.r, MIN(pixel.g, pixel.b)));
        }
    }

    return min;
    
}

// computes the pixels for the dark channel of the image
pixel_t *compute_dark_channel(image_t *image, int patch_size) {
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
            min = find_patch_min(i, j, image, patch_size);

            pixel.r = min;
            pixel.g = min;
            pixel.b = min;
            pixel.a = image->pixels[i * image->width + j].a;

            dark_channel[i * image->width + j] = pixel;
        }
    }

    return dark_channel;
}

// finds the num_pixels brightest pixels in the dark channel and returns their indices
unsigned int *find_brightest_pixels(unsigned int num_pixels, unsigned int height, unsigned int width, pixel_t *dark_channel) {
    float min;
    unsigned int len;
    unsigned int min_index;
    unsigned int i, j;
    unsigned int *indices;
    pixel_t pixel, temp_pixel;

    // indices of pixels in the haze opaque region
    indices = calloc(num_pixels, sizeof(unsigned int));
    if (!indices) {
        return NULL;
    }

    len = 0;
    min_index = 0;
    min = dark_channel[min_index].r;
    for (i = 0; i < height * width; i++) {
        pixel = dark_channel[i];

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
                    temp_pixel = dark_channel[indices[j]];
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

// returns the mean of the pixel rgb values
float pixel_intensity(pixel_t pixel) {
    return (pixel.r + pixel.g + pixel.b) / 3;
}

// finds the brightest pixel in the image from the set of indices
unsigned int find_brightest_pixel(unsigned int *indices, unsigned int num_pixels, image_t *image) {
    float max;
    float temp;
    unsigned int i;
    unsigned int max_index;

    max_index = 0;
    max = pixel_intensity(image->pixels[indices[max_index]]);
    for (i = 1; i < num_pixels; i++) {
        temp = pixel_intensity(image->pixels[indices[i]]);
        if (temp > max) {
            max_index = i;
            max = temp;
        }
    }

    return indices[max_index];
}

// computes an estimate of atmospheric light by finding the brightest pixel in the haze opaque region
void compute_atmospheric_light(pixel_t *atmos_light, image_t *image, pixel_t *dark_channel) {
    unsigned int num_pixels;
    unsigned int index;
    unsigned int *indices;

    // choose the size of the haze opaque region
    num_pixels = image->height * image->width * HAZE_OPAQUE_SIZE;

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
pixel_t *compute_norm_dark_channel(image_t *image, pixel_t *atmos_light, int patch_size) {
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
        norm_pixels[i].a = pixel.a;
    }
    norm_image = replace_pixels(image, norm_pixels);

    norm_dark_channel = compute_dark_channel(norm_image, patch_size);

    free_image(norm_image);

    return norm_dark_channel;
}

// computes the transmission in the image with respect to the atmospheric light
pixel_t *compute_transmission(image_t *image, pixel_t *atmos_light) {
    float temp;
    unsigned int i;
    pixel_t *norm_dark_channel;
    pixel_t *transmission;

    norm_dark_channel = compute_norm_dark_channel(image, atmos_light, TRANSMISSION_PATCH_SIZE);

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
        transmission[i].a = norm_dark_channel[i].a;
    }

    free(norm_dark_channel);

    return transmission;
}

// counts the number of pixels in the windoww
unsigned int count_window_pixels(int y, int x, unsigned int height, unsigned int width, int window_size) {
    unsigned int y_min, y_max, x_min, x_max;

    y_min = MAX(0, y - window_size / 2);
    x_min = MAX(0, x - window_size / 2);
    y_max = MIN(height, y + window_size / 2 + 1);
    x_max = MIN(width, x + window_size / 2 + 1);

    return (y_max - y_min) * (x_max - x_min);
}

// computes the mean for the rgb color channels across the pixels in the window
void compute_window_mean(pixel_t *mean, int y, int x, pixel_t *pixels, unsigned int height, unsigned int width, int window_size) {
    unsigned int y_min, y_max, x_min, x_max;
    unsigned int i, j;
    unsigned int num_pixels;
    pixel_t pixel;

    y_min = MAX(0, y - window_size / 2);
    x_min = MAX(0, x - window_size / 2);
    y_max = MIN(height, y + window_size / 2 + 1);
    x_max = MIN(width, x + window_size / 2 + 1);

    mean->r = 0;
    mean->g = 0;
    mean->b = 0;
    for (i = y_min; i < y_max; i++) {
        for (j = x_min; j < x_max; j++) {
            pixel = pixels[i * width + j];

            mean->r += pixel.r;
            mean->g += pixel.g;
            mean->b += pixel.b;
            mean->a = pixel.a;
        }
    }

    num_pixels = count_window_pixels(y, x, height, width, window_size);

    mean->r /= num_pixels;
    mean->g /= num_pixels;
    mean->b /= num_pixels;
}

// computes the variance for the rgb color channels across the pixels in the window
void compute_window_variance(pixel_t *mean, pixel_t *variance, int y, int x, pixel_t *pixels, unsigned int height, unsigned int width, int window_size) {
    unsigned int y_min, y_max, x_min, x_max;
    unsigned int i, j;
    unsigned int num_pixels;
    pixel_t pixel;

    y_min = MAX(0, y - window_size / 2);
    x_min = MAX(0, x - window_size / 2);
    y_max = MIN(height, y + window_size / 2 + 1);
    x_max = MIN(width, x + window_size / 2 + 1);

    variance->r = 0;
    variance->g = 0;
    variance->b = 0;
    for (i = y_min; i < y_max; i++) {
        for (j = x_min; j < x_max; j++) {
            pixel = pixels[i * width + j];

            variance->r += (pixel.r - mean->r) * (pixel.r - mean->r);
            variance->g += (pixel.g - mean->g) * (pixel.g - mean->g);
            variance->b += (pixel.b - mean->b) * (pixel.b - mean->b);
            variance->a = pixel.a;
        }
    }

    num_pixels = count_window_pixels(y, x, height, width, window_size);

    variance->r /= (num_pixels - 1);
    variance->g /= (num_pixels - 1);
    variance->b /= (num_pixels - 1);
}

// computes the dot product for a window across two sets of pixels
void compute_window_dot_product(pixel_t *dot_product, int y, int x, pixel_t *pixels_X, pixel_t *pixels_Y, unsigned int height, unsigned int width, int window_size) {
    unsigned int y_min, y_max, x_min, x_max;
    unsigned int i, j;
    pixel_t pixel_x, pixel_y;

    y_min = MAX(0, y - window_size / 2);
    x_min = MAX(0, x - window_size / 2);
    y_max = MIN(height, y + window_size / 2 + 1);
    x_max = MIN(width, x + window_size / 2 + 1);

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
            dot_product->a = pixel_x.a;
        }
    }
}

// updates smooth transmission pixels in the window in place
void update_smooth_transmission(int y, int x, pixel_t *smooth_transmission, pixel_t *a, pixel_t *b, image_t *image, int window_size) {
    unsigned int y_min, y_max, x_min, x_max;
    unsigned int i, j;
    pixel_t pixel_image, pixel_transmission;

    y_min = MAX(0, y - window_size / 2);
    x_min = MAX(0, x - window_size / 2);
    y_max = MIN(image->height, y + window_size / 2 + 1);
    x_max = MIN(image->width, x + window_size / 2 + 1);

    for (i = y_min; i < y_max; i++) {
        for (j = x_min; j < x_max; j++) {
            pixel_transmission = smooth_transmission[i * image->width + j];
            pixel_image = image->pixels[i * image->width + j];

            pixel_transmission.r += a->r * pixel_image.r + b->r;
            pixel_transmission.g += a->g * pixel_image.g + b->g;
            pixel_transmission.b += a->b * pixel_image.b + b->b;
            pixel_transmission.a = pixel_image.a;

            smooth_transmission[i * image->width + j] = pixel_transmission;
        }
    }
}

// computes a smooth transmission by reducing blockiness and introducing edges from original image
pixel_t *compute_smooth_transmission(image_t *image, pixel_t *transmission, int window_size) {
    unsigned int i, j;
    unsigned int num_pixels;
    pixel_t a, b;
    pixel_t image_mean, transmission_mean, image_variance, dot_product;
    pixel_t pixel;
    pixel_t *smooth_transmission;

    smooth_transmission = calloc(image->height * image->width, sizeof(pixel_t));
    if (!smooth_transmission) {
        return NULL;
    }

    // compute smooth transmission pixels in each window
    for (i = 0; i < image->height; i++) {
        for (j = 0; j < image->width; j++) {
            compute_window_mean(&image_mean, i, j, image->pixels, image->height, image->width, window_size);
            compute_window_mean(&transmission_mean, i, j, transmission, image->height, image->width, window_size);
            compute_window_variance(&image_mean, &image_variance, i, j, image->pixels, image->height, image->width, window_size);
            compute_window_dot_product(&dot_product, i, j, image->pixels, transmission, image->height, image->width, window_size);
            num_pixels = count_window_pixels(i, j, image->height, image->width, window_size);

            // compute a
            a.r = ((dot_product.r / num_pixels) - (image_mean.r * transmission_mean.r)) / (image_variance.r + EDGE_VARIANCE);
            a.g = ((dot_product.g / num_pixels) - (image_mean.g * transmission_mean.g)) / (image_variance.g + EDGE_VARIANCE);
            a.b = ((dot_product.b / num_pixels) - (image_mean.b * transmission_mean.b)) / (image_variance.b + EDGE_VARIANCE);

            // compute b
            b.r = transmission_mean.r - a.r * image_mean.r;
            b.g = transmission_mean.g - a.g * image_mean.g;
            b.b = transmission_mean.b - a.b * image_mean.b;

            update_smooth_transmission(i, j, smooth_transmission, &a, &b, image, window_size);
        }
    }

    // normalize smooth transmission pixels
    for (i = 0; i < image->height; i++) {
        for (j = 0; j < image->width; j++) {
            pixel = smooth_transmission[i * image->width + j];
            num_pixels = count_window_pixels(i, j, image->height, image->width, window_size);

            pixel.r /= num_pixels;
            pixel.g /= num_pixels;
            pixel.b /= num_pixels;

            smooth_transmission[i * image->width + j] = pixel;
        }
    }

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

        scene_radiance[i].r = (pixel.r - atmos_light->r) / MAX(transmission[i].r, MIN_TRANSMISSION) + atmos_light->r;
        scene_radiance[i].g = (pixel.g - atmos_light->g) / MAX(transmission[i].g, MIN_TRANSMISSION) + atmos_light->g;
        scene_radiance[i].b = (pixel.b - atmos_light->b) / MAX(transmission[i].b, MIN_TRANSMISSION) + atmos_light->b;
        scene_radiance[i].a = pixel.a;
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

    dark_channel = compute_dark_channel(image, DARK_CHANNEL_PATCH_SIZE);
    if (!dark_channel) {
        fprintf(stderr, "Error computing dark channel\n");
        return NULL;
    }

    atmos_light.a = 0;
    compute_atmospheric_light(&atmos_light, image, dark_channel);
    if (!atmos_light.a) {
        fprintf(stderr, "Error estimating atmospheric light\n");
        free(dark_channel);
        return NULL;
    }

    transmission = compute_transmission(image, &atmos_light);
    if (!transmission) {
        fprintf(stderr, "Error computing transmission\n");
        free(dark_channel);
        return NULL;
    }

    smooth_transmission = compute_smooth_transmission(image, transmission, WINDOW_SIZE);
    if (!smooth_transmission) {
        fprintf(stderr, "Error computing smooth transmission\n");
        free(dark_channel);
        free(transmission);
        return NULL;
    }

    scene_radiance = compute_scene_radiance(image, &atmos_light, smooth_transmission);

    dehazed_image = replace_pixels(image, scene_radiance);

    free(dark_channel);
    free(transmission);
    free(smooth_transmission);

    return dehazed_image;
}

// reads image, removes haze, writes resultant image
int main(int argc, char **argv) {
    int opt;
    char *input;
    char *output;
    char *usage = "Usage: %s -i <input_filename> -o <output_filename>\n";
    image_t *image;
    image_t *new_image;

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

    fprintf(stdout, "Parsed %ux%u image\n", image->width, image->height);

    new_image = remove_haze(image);
    if (!new_image) {
        free_image(image);
        return 1;
    }

    fprintf(stdout, "Writing Image to %s\n", output); 

    write_image(new_image, output);

    free_image(image);
    free_image(new_image);

    return 0;
}
