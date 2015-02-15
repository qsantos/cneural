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

#define N_OUTPUTS (10)

void import_case(mnist_t* mnist, float* input, float* expect)
{
    // get data
    unsigned char image[mnist->n_pixels];
    unsigned int label = mnist_next(mnist, image);

    // set input
    for (size_t i = 0; i < mnist->n_pixels; i++)
        input[i] = image[i] / 256.f;

    // set expected output
    for (size_t i = 0; i < N_OUTPUTS; i++)
        expect[i] = 0.f;
    expect[label] = 1.f;
}

int main()
{
    srand(time(NULL));

    neural_network_t nn;
    neural_network_init(&nn, 28*28);
    neural_network_add_layer(&nn, 300);
    neural_network_add_layer(&nn, N_OUTPUTS);

    // train
    size_t n_iterations = 1;
    for (size_t k = 0; k < n_iterations; k++)
    {
        mnist_t mnist;
        mnist_init(&mnist, "mnist/train-labels-idx1-ubyte", "mnist/train-images-idx3-ubyte");
        for (size_t i = 0; i < mnist.n_elements; i++)
        {
            // get case
            float input[mnist.n_pixels];
            float expect[N_OUTPUTS];
            import_case(&mnist, input, expect);

            // train
            neural_network_train(&nn, input, expect);
        }
        nn.learning_rate -= 1.f / n_iterations;
        mnist_exit(&mnist);
    }

    // test
    mnist_t mnist;
    mnist_init(&mnist, "mnist/t10k-labels-idx1-ubyte", "mnist/t10k-images-idx3-ubyte");
    size_t n_tests[N_OUTPUTS] = {0};
    size_t n_successes[N_OUTPUTS] = {0};
    for (size_t i = 0; i < mnist.n_elements; i++)
    {
        // get case
        float input[mnist.n_pixels];
        float expect[N_OUTPUTS];
        import_case(&mnist, input, expect);

        // test
        float output[N_OUTPUTS];
        neural_network_compute(&nn, input, output);

        // retrieve original label
        int label = 0;
        for (; expect[label] != 1.f; label++);

        n_tests[label]++;

        // check result
        int ok = 1;
        for (size_t i = 0; i < N_OUTPUTS; i++)
        {
            if ((output[i] > 0.5f) != (expect[i] > 0.5f))
            {
                ok = 0;
                break;
            }
        }
        n_successes[label] += ok;
    }

    size_t total_successes = 0;
    size_t total_tests = 0;
    for (size_t i = 0; i < N_OUTPUTS; i++)
    {
        size_t s = n_successes[i];
        size_t t = n_tests[i];
        printf("%zu: %4zu / %4zu → %5.2f%%\n", i, s, t, 100.f * s / (float)t);
        total_successes += s;
        total_tests += t;
    }

    size_t s = total_successes;
    size_t t = total_tests;
    printf("\n");
    printf("T: %4zu /%5zu → %5.2f%%\n", s, t, 100.f * s / (float)t);

    mnist_exit(&mnist);

    neural_network_exit(&nn);
    return 0;
}
