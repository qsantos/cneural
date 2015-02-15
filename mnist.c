/*\
 *  C implementation of a neural network
 *  Copyright (C) 2015  Quentin SANTOS
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
\*/

#include "mnist.h"

#include <stdlib.h>

#define CHECK(COND, ...) do { \
        if (!(COND)) { \
            fprintf(stderr, "%s:%u ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__); \
            exit(1); \
        } \
    } while (0)

static unsigned char read_byte(FILE* f)
{
    unsigned char ret;
    fread(&ret, 1, 1, f);
    return ret;
}

static unsigned int read_word(FILE* f)
{
    return 0
        | (read_byte(f) << 24)
        | (read_byte(f) << 16)
        | (read_byte(f) <<  8)
        | (read_byte(f) <<  0)
    ;
}

void mnist_init(mnist_t* mnist, const char* file_labels, const char* file_images)
{
    // open file with label
    mnist->labels = fopen(file_labels, "r");
    CHECK(mnist->labels != NULL, "Could not open file '%s'\n", file_labels);

    // open file with images
    mnist->images = fopen(file_images, "r");
    CHECK(mnist->images != NULL, "Could not open file '%s'\n", file_images);

    // read headers
    unsigned int magicword1 = read_word(mnist->labels);
    CHECK(magicword1 == 2049, "Invalid magic word %u\n", magicword1);
    unsigned int n_labels = read_word(mnist->labels);

    unsigned int magicword2 = read_word(mnist->images);
    CHECK(magicword2 == 2051, "Invalid magic word %u\n", magicword2);
    unsigned int n_images = read_word(mnist->images);
    unsigned int n_rows = read_word(mnist->images);
    unsigned int n_cols = read_word(mnist->images);

    // save file structure information
    CHECK(n_labels == n_images, "Numbers of labels (%u) and images (%u) mismatch\n", n_labels, n_images);
    mnist->n_elements = n_labels;
    mnist->n_pixels = n_rows * n_cols;
}

void mnist_exit(mnist_t* mnist)
{
    fclose(mnist->images);
    fclose(mnist->labels);
}

unsigned int mnist_next(mnist_t* mnist, unsigned char* image)
{
    fread(image, 1, mnist->n_pixels, mnist->images);
    return read_byte(mnist->labels);
}
