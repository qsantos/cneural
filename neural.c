#include <stdlib.h>
#include <stdio.h>
#include <math.h>

float sigmoid(float x)
{
    return 1.f / (1.f + expf(-x));
}

float sigmoid_prime(float x)
{
    float s = sigmoid(x);
    return (1 - s) * s;
}

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
