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

#ifndef NETWORK_H
#define NETWORK_H

#include "neuron.h"

typedef struct neural_network neural_network_t;

struct neural_network
{
    size_t n_neurons;
    neuron_t* neurons;

    size_t n_inputs;
    float learning_rate;

    size_t layer_size;
};

void neural_network_init(neural_network_t* nn, size_t n_inputs);
void neural_network_exit(neural_network_t* nn);

void neural_network_add_layer(neural_network_t* nn, size_t n_neurons);

float neural_network_propagate    (neural_network_t* nn);
void  neural_network_backpropagate(neural_network_t* nn, float gradient);

void neural_network_input(neural_network_t* nn, float* input);

#endif
