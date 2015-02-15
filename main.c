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

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "network.h"
#include "mnist.h"

void bytes2floats(size_t n, unsigned char* b, float* f)
{
    for (size_t i = 0; i < n; i++)
        f[i] = b[i] / 256.f;
}

int main()
{
    srand(time(NULL));

    neural_network_t nn;
    neural_network_init(&nn, 28*28);
    neural_network_add_layer(&nn, 300);
    neural_network_add_layer(&nn, 1);

    // train
    size_t n_iterations = 1;
    for (size_t k = 0; k < n_iterations; k++)
    {
        mnist_t mnist;
        mnist_init(&mnist, "mnist/train-labels-idx1-ubyte", "mnist/train-images-idx3-ubyte");
        unsigned char image[mnist.n_pixels];
        float input[mnist.n_pixels];
        for (size_t i = 0; i < mnist.n_elements; i++)
        {
            // get image
            unsigned int label = mnist_next(&mnist, image);
            bytes2floats(mnist.n_pixels, image, input);

            // differentiate zeros
            float output = label == 0;

            // train
            neural_network_input(&nn, input);
            float result = neural_network_propagate(&nn);
            neural_network_backpropagate(&nn, result - output);
        }
        nn.learning_rate -= 1.f / n_iterations;
        mnist_exit(&mnist);
    }

    // test
    mnist_t mnist;
    mnist_init(&mnist, "mnist/t10k-labels-idx1-ubyte", "mnist/t10k-images-idx3-ubyte");
    size_t n_tests = 0;
    size_t n_successes = 0;
    unsigned char image[mnist.n_pixels];
    float input[mnist.n_pixels];
    for (size_t i = 0; i < mnist.n_elements; i++)
    {
        // get image
        unsigned int label = mnist_next(&mnist, image);
        bytes2floats(mnist.n_pixels, image, input);

        // differentiate zeros
        float output = label == 0;

        // test
        neural_network_input(&nn, input);
        float result = neural_network_propagate(&nn);

        if ((result > 0.5f) == (output > 0.5f))
            n_successes++;

        n_tests++;
    }
    printf("%zu / %zu\n", n_successes, n_tests);
    mnist_exit(&mnist);

    neural_network_exit(&nn);
    return 0;
}
