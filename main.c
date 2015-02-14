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

int main()
{
    neural_network_t nn;
    neural_network_init(&nn, 2);
    neural_network_add_layer(&nn, 1);

    // train
    for (int i = 0; i < 100000; i++)
    {
        int a = rand() & 1;
        int b = rand() & 1;
        int output = a | b;

        nn.neurons[1].output = (float)a;
        nn.neurons[2].output = (float)b;

        neural_network_propagate(&nn);
        nn.neurons[3].local_gradient = nn.neurons[3].output - output;
        neural_network_backpropagate(&nn);
    }

    // test
    for (int a = 0; a <= 1; a++)
        for (int b = 0; b <= 1; b++)
        {
            int output = a | b;
            nn.neurons[1].output = (float)a;
            nn.neurons[2].output = (float)b;

            neural_network_propagate(&nn);

            float result = nn.neurons[3].output;
            printf("%i %f\n", output, result);
        }

    neural_network_exit(&nn);
    return 0;
}
