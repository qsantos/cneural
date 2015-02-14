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

float weights[3];

typedef struct neuron neuron_t;

struct neuron
{
    float local_field;
    float local_gradient;
    float output;
};

neuron_t neurons[4];

float neuron_propagate(neuron_t* neuron)
{
    // compute local field δ_i = sum(y_j w_ij)
    float local_field = 0;
    for (size_t i = 0; i < 3; i++)
        local_field += neurons[i].output * weights[i];
    neuron->local_field = local_field;

    // compute output y_i
    float output = sigmoid(local_field);
    neuron->output = output;

    return output;
}

void neuron_backpropagate(neuron_t* neuron)
{
    float local_gradient = neuron->local_gradient * sigmoid_prime(neuron->local_field);

    for (size_t i = 0; i < 3; i++)
        weights[i] -= local_gradient * neurons[i].output;
}

int main()
{
    neurons[0].output = 1.f;

    neuron_t* neuron = &neurons[3];

    for (int i = 0; i < 100000; i++)
    {
        int a = rand() & 1;
        int b = rand() & 1;
        int output = a | b;

        neurons[1].output = (float)a;
        neurons[2].output = (float)b;

        float result = neuron_propagate(neuron);

        float correction = result - output;
        neuron->local_gradient = correction;
        neuron_backpropagate(neuron);
    }

    for (int a = 0; a <= 1; a++)
        for (int b = 0; b <= 1; b++)
        {
            int output = a | b;
            neurons[1].output = (float)a;
            neurons[2].output = (float)b;
            float result = neuron_propagate(neuron);
            printf("%i %f\n", output, result);
        }

    return 0;
}
