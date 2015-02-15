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

void import_case(mnist_t* mnist, float* input, float* expect)
{
    unsigned char image[mnist->n_pixels];
    unsigned int label = mnist_next(mnist, image);
    for (size_t i = 0; i < mnist->n_pixels; i++)
        input[i] = image[i] / 256.f;

    expect[0] = label == 0; // differentiate zeros
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
        for (size_t i = 0; i < mnist.n_elements; i++)
        {
            // get case
            float input[mnist.n_pixels];
            float expect[1];
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
    size_t n_tests = 0;
    size_t n_successes = 0;
    for (size_t i = 0; i < mnist.n_elements; i++)
    {
        // get case
        float input[mnist.n_pixels];
        float expect[1];
        import_case(&mnist, input, expect);

        // test
        float output[1];
        neural_network_compute(&nn, input, output);
        if ((output[0] > 0.5f) == (expect[0] > 0.5f))
            n_successes++;

        n_tests++;
    }
    printf("%zu / %zu\n", n_successes, n_tests);
    mnist_exit(&mnist);

    neural_network_exit(&nn);
    return 0;
}
