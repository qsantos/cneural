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

#ifndef NEURON_H
#define NEURON_H

#include <stddef.h>

typedef struct synapse synapse_t;
typedef struct neuron neuron_t;

struct synapse
{
    size_t neighbour_index;
    float weight;
};

struct neuron
{
    float local_field;
    float local_gradient;
    float output;

    size_t n_synapses;
    synapse_t* synapses;
};

float neuron_propagate    (neuron_t* neurons, size_t i);
void  neuron_backpropagate(neuron_t* neurons, size_t i);

#endif
