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

void neural_network_init(neural_network_t* nn, size_t n_inputs)
{
    size_t n_neurons = 1 + n_inputs;

    nn->n_neurons = n_neurons;
    nn->neurons = calloc(n_neurons, sizeof(neuron_t));

    // normal input, for neuron bias
    nn->neurons[0].output = 1.f;

    nn->n_inputs = n_inputs;
    nn->learning_rate = 1.f;
    nn->layer_size = n_inputs;
}

void neural_network_exit(neural_network_t* nn)
{
    for (size_t i = 0; i < nn->n_neurons; i++)
        free(nn->neurons[i].synapses);
    free(nn->neurons);
}

void neural_network_add_layer(neural_network_t* nn, size_t n_neurons)
{
    size_t last_neuron = nn->n_neurons;

    nn->n_neurons += n_neurons;
    nn->neurons = realloc(nn->neurons, nn->n_neurons * sizeof(neuron_t));

    size_t layer_start = last_neuron - nn->layer_size;
    for (size_t i = last_neuron; i < nn->n_neurons; i++)
    {
        neuron_t* neuron = &nn->neurons[i];

        size_t n_synapses = 1 + nn->layer_size;
        neuron->n_synapses = n_synapses;

        neuron->synapses = calloc(n_synapses, sizeof(synapse_t));
        neuron->synapses[0].neighbour_index = 0; // bias
        for (size_t j = 0; j < nn->layer_size; j++)
        {
            neuron->synapses[1+j].neighbour_index = layer_start + j;
            neuron->synapses[1+j].weight = 2.f * (float)rand() / RAND_MAX - 1;
        }
    }

    nn->layer_size = n_neurons;
}

float neural_network_propagate(neural_network_t* nn)
{
    for (size_t i = 1 + nn->n_inputs; i < nn->n_neurons; i++)
        neuron_propagate(nn->neurons, i);
    return nn->neurons[nn->n_neurons-1].output;
}

void neural_network_backpropagate(neural_network_t* nn, float gradient)
{
    nn->neurons[nn->n_neurons-1].local_gradient = gradient;
    for (size_t i = nn->n_neurons; i-- > 1 + nn->n_inputs; )
        neuron_backpropagate(nn->neurons, i, nn->learning_rate);
}

void neural_network_input(neural_network_t* nn, float* input)
{
    for (size_t i = 1; i < 1 + nn->n_inputs; i++, input++)
        nn->neurons[i].output = *input;
}
