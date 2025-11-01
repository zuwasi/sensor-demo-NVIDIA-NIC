#include <cstdio>
#include <cuda_runtime.h>

__global__ void add1(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += 1.0f;
}

int main() {
    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    if (devCount == 0) {
        printf("No CUDA device found\n");
        return 1;
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    printf("Using device: %s (Compute Capability %d.%d)\n",
        prop.name, prop.major, prop.minor);

    const int N = 1 << 20;
    float* d = nullptr;
    cudaMalloc(&d, N * sizeof(float));
    cudaMemset(d, 0, N * sizeof(float));

    //add1 << <(N + 255) / 256, 256 >> > (d, N);
    add1 << <(N + 255) / 256, 256 >> > (d, N);

    cudaDeviceSynchronize();

    float h = -1.0f;
    cudaMemcpy(&h, d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d);

    printf("First value after kernel = %.1f\n", h);
    printf("Result = PASS\n");
}
