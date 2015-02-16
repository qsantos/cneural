#include <stdio.h>
#include <time.h>
#include <curand_kernel.h>

extern "C"
{
#include "mnist.h"
}

#define n_inputs (28*28)
#define n_outputs (10)

__shared__ float weights[n_outputs][n_inputs+1];

__device__ float sigmoid(float x)
{
    return 1.f / (1.f + expf(-0.007f * x));
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
    float local_field = weights[i][n_inputs];
    for (size_t j = 0; j < n_inputs; j++)
        local_field += inputs[j] * weights[i][j];

    // compute outputs, y_i = ϕ(v_i)
    outputs[i] = sigmoid(local_field);
}

__device__ void train(float* inputs, float* expect)
{
    const int i = threadIdx.x;

    __shared__ float outputs[n_outputs];
    compute(inputs, outputs);
    float local_gradient = outputs[i] - expect[i];

    // update weights
    weights[i][n_inputs] -= local_gradient;
    for (size_t j = 0; j < n_inputs; j++)
        weights[i][j] -= local_gradient * inputs[j];
}

__global__ void init(int seed)
{
    const int i = threadIdx.x;
    curandState_t state;
    curand_init(seed, i, 0, &state);
    for (size_t j = 0; j < n_inputs; j++)
        weights[i][j] = 2.f * curand_uniform(&state) - 1.f;
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
    for (size_t i = 0; i < n_outputs; i++)
        expect[i] = 0.f;
    expect[label] = 1.f;
}

int main()
{
    float hinputs[n_inputs];
    float hexpect[n_outputs];
    float houtput[n_outputs];

    float* dinputs; cudaMalloc((void**)&dinputs,  n_inputs*sizeof(float));
    float* dexpect; cudaMalloc((void**)&dexpect, n_outputs*sizeof(float));
    float* doutput; cudaMalloc((void**)&doutput, n_outputs*sizeof(float));

    size_t n_nodes = n_outputs;
    init<<<1, n_nodes>>>(time(NULL));

    printf("training\n");
    mnist_t mnist;
    mnist_init(&mnist, "mnist/train-labels-idx1-ubyte", "mnist/train-images-idx3-ubyte");
    for (int i = 0; i < mnist.n_elements; i++)
    {
        import_case(&mnist, hinputs, hexpect);

        cudaMemcpy(dinputs, hinputs,  n_inputs*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dexpect, hexpect, n_outputs*sizeof(float), cudaMemcpyHostToDevice);
        do_train<<<1, n_nodes>>>(dinputs, dexpect);
    }
    mnist_exit(&mnist);

    printf("testing\n");
    mnist_init(&mnist, "mnist/t10k-labels-idx1-ubyte", "mnist/t10k-images-idx3-ubyte");
    size_t classifications[n_outputs][n_outputs] = {{0}};

    for (int i = 0; i < mnist.n_elements; i++)
    {
        import_case(&mnist, hinputs, hexpect);
        cudaMemcpy(dinputs, hinputs,  n_inputs*sizeof(float), cudaMemcpyHostToDevice);
        do_compute<<<1, n_nodes>>>(dinputs, doutput);
        cudaMemcpy(houtput, doutput, n_outputs*sizeof(float), cudaMemcpyDeviceToHost);

        // retrieve original label
        int label = 0;
        for (; hexpect[label] != 1.f; label++);

        // get result
        float best = 0;
        int selected = 0;
        for (size_t i = 0; i < n_outputs; i++)
        {
            if (houtput[i] > best)
            {
                best = houtput[i];
                selected = i;
            }
        }

        // log classification
        classifications[label][selected]++;
    }

    // table header
    printf("    ");
    for (size_t j = 0; j < n_outputs; j++)
        printf("%4zu ", j);
    printf(" total\n");
    printf("\n");

    for (size_t i = 0; i < n_outputs; i++)
    {
        size_t total = 0;
        printf("%zu   ", i);
        for (size_t j = 0; j < n_outputs; j++)
        {
            printf("%4zu ", classifications[i][j]);
            total += classifications[i][j];
        }
        printf("  %4zu\n", total);
    }

    printf("\n");
    printf("tot ");
    for (size_t j = 0; j < n_outputs; j++)
    {
        size_t total = 0;
        for (size_t i = 0; i < n_outputs; i++)
            total += classifications[i][j];
        printf("%4zu ", total);
    }
    printf("\n");

    printf("\n");
    printf("Caption: image with digit LINE was classified as a COLUMN digit CELL times\n");

    printf("\n");
    size_t correct = 0;
    size_t total = 0;
    for (size_t i = 0; i < n_outputs; i++)
    {
        for (size_t j = 0; j < n_outputs; j++)
            total += classifications[i][j];
        correct += classifications[i][i];
    }
    printf("%zu / %zu → %5.2f\n", correct, total, 100.f*correct/(float)total);


    cudaFree(doutput);
    cudaFree(dexpect);
    cudaFree(dinputs);
    return 0;
}
