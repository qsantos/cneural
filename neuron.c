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

#include "neuron.h"

#include <math.h>

static float sigmoid(float x)
{
    return 1.f / (1.f + expf(-x));
}

static float sigmoid_prime(float x)
{
    float s = sigmoid(x);
    return (1 - s) * s;
}

float neuron_propagate(neuron_t* neurons, size_t i)
{
    neuron_t* neuron = &neurons[i];

    // compute local field, v_i = sum(y_j w_ji)
    float local_field = 0;
    for (size_t j = 0; j < neuron->n_synapses; j++)
    {
        synapse_t* s = &neuron->synapses[j];
        local_field += neurons[s->neighbour_index].output * s->weight;
    }
    neuron->local_field = local_field;

    // compute output, y_i
    float output = sigmoid(local_field);
    neuron->output = output;

    return output;
}

void neuron_backpropagate(neuron_t* neurons, size_t i)
{
    neuron_t* neuron = &neurons[i];

    // finalize computation of local gradient, δ_i = ϕ'(v_i) × …
    float local_gradient = neuron->local_gradient * sigmoid_prime(neuron->local_field);

    // update weights
    for (size_t j = 0; j < neuron->n_synapses; j++)
    {
        synapse_t* s = &neuron->synapses[j];
        s->weight -= local_gradient * neurons[s->neighbour_index].output;
    }
}
