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

int main()
{
    srand(time(NULL));

    neural_network_t nn;
    neural_network_init(&nn, 28*28);
    neural_network_add_layer(&nn, 100);
    neural_network_add_layer(&nn, 1);

    // train
    size_t n_subjects = 300;
    size_t n_iterations = 20;
    for (size_t k = 0; k < n_iterations; k++)
    {
        mnist_t mnist;
        mnist_init(&mnist, "mnist/train-labels-idx1-ubyte", "mnist/train-images-idx3-ubyte");
        unsigned char image[mnist.n_pixels];
        size_t subjects = 0;
        for (size_t i = 0; i < mnist.n_elements; i++)
        {
            unsigned int label = mnist_next(&mnist, image);

            // try to differentiate zeros from fives
            float output;
            if (label == 0)
                output = 1.f;
            else if (label == 5)
                output = 0.f;
            else
                continue;

            neural_network_input_from_bytes(&nn, image);
            float result = neural_network_propagate(&nn);
            neural_network_backpropagate(&nn, result - output);

            if (++subjects >= n_subjects)
                break;
        }
        mnist_exit(&mnist);
    }

    // test
    mnist_t mnist;
    mnist_init(&mnist, "mnist/train-labels-idx1-ubyte", "mnist/train-images-idx3-ubyte");
    size_t n_tests = 0;
    size_t n_successes = 0;
    unsigned char image[mnist.n_pixels];
    for (size_t i = 0; i < mnist.n_elements; i++)
    {
        unsigned int label = mnist_next(&mnist, image);

        float output;
        if (label == 0)
            output = 1.f;
        else if (label == 5)
            output = 0.f;
        else
            continue;

        neural_network_input_from_bytes(&nn, image);
        float result = neural_network_propagate(&nn);

        if ((result > 0.5f) == (output > 0.5f))
            n_successes++;

        if (++n_tests >= n_subjects)
            break;
    }
    printf("%zu / %zu\n", n_successes, n_tests);
    mnist_exit(&mnist);

    neural_network_exit(&nn);
    return 0;
}
