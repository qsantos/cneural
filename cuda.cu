#include <stdio.h>

extern "C"
{
#include "mnist.h"
}

#define n_inputs (28*28)

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

void import_case(mnist_t* mnist, float* input, float* expect)
{
    // get data
    unsigned char image[mnist->n_pixels];
    unsigned int label = mnist_next(mnist, image);

    // set input
    for (size_t i = 0; i < mnist->n_pixels; i++)
        input[i] = image[i] / 256.f;

    // set expected output
    expect[0] = (float)(label == 0);
}

int main()
{

    float hinputs[n_inputs];
    float hexpect[1];
    float houtput[1];

    float* dinputs; cudaMalloc((void**)&dinputs, n_inputs*sizeof(float));
    float* dexpect; cudaMalloc((void**)&dexpect,        1*sizeof(float));
    float* doutput; cudaMalloc((void**)&doutput,        1*sizeof(float));

    mnist_t mnist;
    mnist_init(&mnist, "mnist/train-labels-idx1-ubyte", "mnist/train-images-idx3-ubyte");
    for (int i = 0; i < mnist.n_elements; i++)
    {
        import_case(&mnist, hinputs, hexpect);

        cudaMemcpy(dinputs, hinputs, n_inputs*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dexpect, hexpect,        1*sizeof(float), cudaMemcpyHostToDevice);
        do_train<<<1, 1>>>(dinputs, dexpect);
    }
    mnist_exit(&mnist);

    mnist_init(&mnist, "mnist/t10k-labels-idx1-ubyte", "mnist/t10k-images-idx3-ubyte");
    size_t correct = 0;
    for (int i = 0; i < mnist.n_elements; i++)
    {
        import_case(&mnist, hinputs, hexpect);
        cudaMemcpy(dinputs, hinputs, n_inputs*sizeof(float), cudaMemcpyHostToDevice);
        do_compute<<<1, 1>>>(dinputs, doutput);
        cudaMemcpy(houtput, doutput,        1*sizeof(float), cudaMemcpyDeviceToHost);

        if ((hexpect[0] > 0.5f) == (houtput[0] > 0.5f))
            correct++;
    }
    printf("%zu / %zu\n", correct, mnist.n_elements);
    mnist_exit(&mnist);

    cudaFree(doutput);
    cudaFree(dexpect);
    cudaFree(dinputs);
    return 0;
}
