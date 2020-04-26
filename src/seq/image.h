/*
 * All image-related operations.
 */

// maximum value for rgba channels
#define PIXEL_MAX_VALUE 255

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
image_t *init_image(unsigned int height, unsigned int width, unsigned char *bytes);

// frees an image
void free_image(image_t *image);

// reads an image from file
image_t *read_image(char *filename);

// writes the image to file
void write_image(image_t *image, char *filename);

// returns a copy of the image after replacing pixels
image_t *replace_pixels(image_t *image, pixel_t *pixels);

// converts all pixels into a single array in rgba format
unsigned char *collapse_pixels(image_t *image);

// returns the mean of the pixel rgb values
float pixel_intensity(pixel_t *pixel);

// finds the num_pixels brightest pixels in the given set and returns their indices
unsigned int *find_brightest_pixels(unsigned int num_pixels, pixel_t *pixels, unsigned int height, unsigned int width);

// finds the brightest pixel in the image from the set of indices and returns its index
unsigned int find_brightest_pixel(image_t *image, unsigned int *indices, unsigned int num_pixels);
