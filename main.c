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

#include "neuron.h"

int main()
{
    neuron_t neurons[4];
    neurons[0].output = 1.f;

    neuron_t* neuron = &neurons[3];

    size_t n_inputs = 2;
    size_t n_synapses = 1 + n_inputs;
    neuron->n_synapses = n_synapses;

    neuron->synapses = calloc(n_synapses, sizeof(synapse_t));
    for (size_t i = 0; i < n_synapses; i++)
        neuron->synapses[i].neighbour_index = i;

    for (int i = 0; i < 100000; i++)
    {
        int a = rand() & 1;
        int b = rand() & 1;
        int output = a | b;

        neurons[1].output = (float)a;
        neurons[2].output = (float)b;

        float result = neuron_propagate(neurons, 3);

        float correction = result - output;
        neuron->local_gradient = correction;
        neuron_backpropagate(neurons, 3);
    }

    for (int a = 0; a <= 1; a++)
        for (int b = 0; b <= 1; b++)
        {
            int output = a | b;
            neurons[1].output = (float)a;
            neurons[2].output = (float)b;
            float result = neuron_propagate(neurons, 3);
            printf("%i %f\n", output, result);
        }

    free(neuron->synapses);
    return 0;
}
