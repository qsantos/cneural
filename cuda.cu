#include <stdio.h>

#define n_inputs 2

__device__ float weights[n_inputs+1];

__device__ float sigmoid(float x)
{
    return 1.f / (1.f + expf(-x));
}

__device__ float sigmoid_prime(float x)
{
    float s = sigmoid(x);
    return (1 - s) * s;
}

__device__ void compute(float* inputs, float* outputs)
{
    const int i = threadIdx.x;

    // compute local field, v_i = sum(y_j w_ji)
    float local_field = weights[n_inputs];
    for (size_t j = 0; j < n_inputs; j++)
        local_field += inputs[j] * weights[j];

    // compute outputs, y_i = ϕ(v_i)
    outputs[i] = sigmoid(local_field);
}

__device__ void train(float* inputs, float* expect)
{
    const int i = threadIdx.x;

    float outputs[1];
    compute(inputs, outputs);
    float local_gradient = outputs[i] - expect[i];

    // update weights
    weights[n_inputs] -= local_gradient;
    for (size_t j = 0; j < n_inputs; j++)
        weights[j] -= local_gradient * inputs[j];
}

__global__ void do_compute(float* inputs, float* outputs)
{
    compute(inputs, outputs);
}

__global__ void do_train(float* inputs, float* outputs)
{
    train(inputs, outputs);
}

int main()
{
    float hinputs[n_inputs], hexpect[1];

    float* dinputs; cudaMalloc((void**)&dinputs, n_inputs*sizeof(float));
    float* dexpect; cudaMalloc((void**)&dexpect,        1*sizeof(float));

    for (int i = 0; i < 200; i++)
    {
        int a = rand() & 1;
        int b = rand() & 1;
        hinputs[0] = a;
        hinputs[1] = b;
        hexpect[0] = a | b;

        cudaMemcpy(dinputs, hinputs, n_inputs*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dexpect, hexpect,        1*sizeof(float), cudaMemcpyHostToDevice);
        do_train<<<1, 1>>>(dinputs, dexpect);
    }

    for (int a = 0; a <= 1; a++)
        for (int b = 0; b <= 1; b++)
        {
            hinputs[0] = a;
            hinputs[1] = b;
            cudaMemcpy(dinputs, hinputs, n_inputs*sizeof(float), cudaMemcpyHostToDevice);
            do_compute<<<1, 1>>>(dinputs, dexpect);
            cudaMemcpy(hexpect, dexpect,        1*sizeof(float), cudaMemcpyDeviceToHost);

            printf("%i %i %f\n", a, b, hexpect[0]);
        }

    cudaFree(dexpect);
    cudaFree(dinputs);
    return 0;
}
