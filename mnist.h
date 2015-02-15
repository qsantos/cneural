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

#ifndef MNIST_H
#define MNIST_H

#include <stdio.h>

typedef struct mnist mnist_t;

struct mnist
{
    FILE* labels;
    FILE* images;

    size_t n_elements;
    size_t n_pixels;
};

void open_labels(mnist_t* mnist, const char* filename);
void open_images(mnist_t* mnist, const char* filename);

void mnist_init(mnist_t* mnist, const char* file_labels, const char* file_images);
void mnist_exit(mnist_t* mnist);

unsigned int mnist_next(mnist_t* mnist, unsigned char* image);

#endif
