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

#include "network.h"

#include <stdlib.h>

void neural_network_init(neural_network_t* nn)
{
    size_t n_neurons = 4;

    nn->n_neurons = n_neurons;
    nn->neurons = calloc(n_neurons, sizeof(neuron_t));

    // normal input, for neuron bias
    nn->neurons[0].output = 1.f;

    nn->n_inputs = 2;
}

void neural_network_exit(neural_network_t* nn)
{
    free(nn->neurons);
}

void neural_network_propagate(neural_network_t* nn)
{
    for (size_t i = 1 + nn->n_inputs; i < nn->n_neurons; i++)
        neuron_propagate(nn->neurons, i);
}

void neural_network_backpropagate(neural_network_t* nn)
{
    for (size_t i = nn->n_neurons; i-- > 1 + nn->n_inputs; )
        neuron_backpropagate(nn->neurons, i);
}
