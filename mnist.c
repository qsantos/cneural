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
#include <stdio.h>

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

void open_labels(const char* filename)
{
    FILE* f = fopen(filename, "r");
    CHECK(f != NULL, "Could not open file %s\n", filename);

    unsigned int magicword = read_word(f);
    CHECK(magicword == 2049, "Invalid magic word %u\n", magicword);

    unsigned int n_labels = read_word(f);
    printf("Found %u labels\n", n_labels);

    unsigned int label = read_byte(f);
    printf("First label is %u\n", label);

    fclose(f);
}

void open_images(const char* filename)
{
    FILE* f = fopen(filename, "r");
    CHECK(f != NULL, "Could not open file %s\n", filename);

    unsigned int magicword = read_word(f);
    CHECK(magicword == 2051, "Invalid magic word %u\n", magicword);

    unsigned int n_images = read_word(f);
    printf("Found %u images\n", n_images);

    unsigned int n_rows = read_word(f);
    unsigned int n_cols = read_word(f);
    printf("Dimensions are %u×%u\n", n_rows, n_cols);

    unsigned char image[n_rows * n_cols];
    fread(image, n_rows*n_cols, 1, f);

    printf("First image is:\n");
    for (unsigned int i = 0; i < n_rows; i++)
    {
        for (unsigned int j = 0; j < n_cols; j++)
        {
            printf("%3u", image[i*n_cols + j]);
        }
        printf("\n");
    }

    fclose(f);
}
