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
    size_t classifications[N_OUTPUTS][N_OUTPUTS] = {{0}};

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

        // get result
        float best = 0;
        int selected = 0;
        for (size_t i = 0; i < N_OUTPUTS; i++)
        {
            if (output[i] > best)
            {
                best = output[i];
                selected = i;
            }
        }

        // log classification
        classifications[label][selected]++;
    }

    // table header
    printf("    ");
    for (size_t j = 0; j < N_OUTPUTS; j++)
        printf("%4zu ", j);
    printf(" total\n");
    printf("\n");

    for (size_t i = 0; i < N_OUTPUTS; i++)
    {
        size_t total = 0;
        printf("%zu   ", i);
        for (size_t j = 0; j < N_OUTPUTS; j++)
        {
            printf("%4zu ", classifications[i][j]);
            total += classifications[i][j];
        }
        printf("  %4zu\n", total);
    }

    printf("\n");
    printf("tot ");
    for (size_t j = 0; j < N_OUTPUTS; j++)
    {
        size_t total = 0;
        for (size_t i = 0; i < N_OUTPUTS; i++)
            total += classifications[i][j];
        printf("%4zu ", total);
    }
    printf("\n");

    printf("\n");
    printf("Caption: image with digit LINE was classified as a COLUMN digit CELL times\n");

    printf("\n");
    size_t correct = 0;
    size_t total = 0;
    for (size_t i = 0; i < N_OUTPUTS; i++)
    {
        for (size_t j = 0; j < N_OUTPUTS; j++)
            total += classifications[i][j];
        correct += classifications[i][i];
    }
    printf("%zu / %zu → %5.2f\n", correct, total, 100.f*correct/(float)total);

    mnist_exit(&mnist);

    neural_network_exit(&nn);
    return 0;
}
