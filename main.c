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

#include "network.h"
#include "mnist.h"

int main()
{
    mnist_t mnist;
    mnist_init(&mnist, "mnist/train-labels-idx1-ubyte", "mnist/train-images-idx3-ubyte");

    unsigned char image[mnist.n_pixels];
    unsigned int label = mnist_next(&mnist, image);
    printf("This is a %u:\n", label);
    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            printf("%3u", image[i*28 + j]);
        }
        printf("\n");
    }

    mnist_exit(&mnist);
    return 0;

    neural_network_t nn;
    neural_network_init(&nn, 2);
    neural_network_add_layer(&nn, 10);
    neural_network_add_layer(&nn, 1);

    // train
    for (int i = 0; i < 100000; i++)
    {
        unsigned char input[2] = {rand()&1, rand()&1};
        neural_network_input_from_bytes(&nn, input);

        char output = input[0] ^ input[1];
        float result = neural_network_propagate(&nn);
        neural_network_backpropagate(&nn, result - output);
    }

    // test
    for (char a = 0; a <= 1; a++)
        for (char b = 0; b <= 1; b++)
        {
            unsigned char input[2] = {a, b};
            neural_network_input_from_bytes(&nn, input);

            char output = input[0] ^ input[1];
            float result = neural_network_propagate(&nn);
            printf("%i %f\n", output, result);
        }

    neural_network_exit(&nn);
    return 0;
}
