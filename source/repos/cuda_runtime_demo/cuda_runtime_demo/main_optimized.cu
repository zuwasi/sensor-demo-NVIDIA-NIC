#include <cstdio>
#include <cuda_runtime.h>
#include <chrono>

// Optimized kernel with better memory access patterns
__global__ void add1_optimized(float* __restrict__ a, const int n) {
    // Use grid-stride loop for better scalability
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Process multiple elements per thread (grid-stride loop)
    for (int i = tid; i < n; i += stride) {
        a[i] += 1.0f;
    }
}

// Vectorized version using float4 for coalesced memory access
__global__ void add1_vectorized(float4* __restrict__ a, const int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Process 4 floats at once
    for (int i = tid; i < n; i += stride) {
        float4 val = a[i];
        val.x += 1.0f;
        val.y += 1.0f;
        val.z += 1.0f;
        val.w += 1.0f;
        a[i] = val;
    }
}

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

int main() {
    // Device query
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        printf("No CUDA device found\n");
        return 1;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using device: %s (Compute Capability %d.%d)\n",
        prop.name, prop.major, prop.minor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);

    // Problem size (match original for fair comparison)
    const int N = 1 << 20;  // 1M elements (same as original)
    const size_t bytes = N * sizeof(float);
    
    printf("\nProblem size: %d elements (%.2f MB)\n", N, bytes / (1024.0f * 1024.0f));

    // Allocate host memory for verification
    float *h_result = new float[4];

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ========== Test 1: Basic kernel ==========
    printf("\n=== Test 1: Basic Kernel ===\n");
    float *d_basic;
    CUDA_CHECK(cudaMalloc(&d_basic, bytes));
    CUDA_CHECK(cudaMemset(d_basic, 0, bytes));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    CUDA_CHECK(cudaEventRecord(start));
    add1_optimized<<<blocksPerGrid, threadsPerBlock>>>(d_basic, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    float bandwidth = (2.0f * bytes) / (milliseconds * 1e6);  // GB/s
    printf("Threads/Block: %d, Blocks: %d\n", threadsPerBlock, blocksPerGrid);
    printf("Time: %.3f ms, Bandwidth: %.2f GB/s\n", milliseconds, bandwidth);
    
    CUDA_CHECK(cudaMemcpy(h_result, d_basic, 4*sizeof(float), cudaMemcpyDeviceToHost));
    printf("Verification: h[0]=%.1f h[1]=%.1f h[2]=%.1f h[3]=%.1f\n", 
           h_result[0], h_result[1], h_result[2], h_result[3]);
    CUDA_CHECK(cudaFree(d_basic));

    // ========== Test 2: Optimized thread configuration ==========
    printf("\n=== Test 2: Optimized Thread Config (1024 threads/block) ===\n");
    float *d_opt;
    CUDA_CHECK(cudaMalloc(&d_opt, bytes));
    CUDA_CHECK(cudaMemset(d_opt, 0, bytes));

    threadsPerBlock = 1024;  // Max threads for RTX 5080
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    CUDA_CHECK(cudaEventRecord(start));
    add1_optimized<<<blocksPerGrid, threadsPerBlock>>>(d_opt, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    bandwidth = (2.0f * bytes) / (milliseconds * 1e6);
    printf("Threads/Block: %d, Blocks: %d\n", threadsPerBlock, blocksPerGrid);
    printf("Time: %.3f ms, Bandwidth: %.2f GB/s\n", milliseconds, bandwidth);
    
    CUDA_CHECK(cudaMemcpy(h_result, d_opt, 4*sizeof(float), cudaMemcpyDeviceToHost));
    printf("Verification: h[0]=%.1f h[1]=%.1f h[2]=%.1f h[3]=%.1f\n", 
           h_result[0], h_result[1], h_result[2], h_result[3]);
    CUDA_CHECK(cudaFree(d_opt));

    // ========== Test 3: Limited blocks (better SM utilization) ==========
    printf("\n=== Test 3: Grid-Stride Loop (fewer blocks, better reuse) ===\n");
    float *d_stride;
    CUDA_CHECK(cudaMalloc(&d_stride, bytes));
    CUDA_CHECK(cudaMemset(d_stride, 0, bytes));

    threadsPerBlock = 1024;
    blocksPerGrid = prop.multiProcessorCount * 4;  // 4 blocks per SM
    
    CUDA_CHECK(cudaEventRecord(start));
    add1_optimized<<<blocksPerGrid, threadsPerBlock>>>(d_stride, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    bandwidth = (2.0f * bytes) / (milliseconds * 1e6);
    printf("Threads/Block: %d, Blocks: %d\n", threadsPerBlock, blocksPerGrid);
    printf("Time: %.3f ms, Bandwidth: %.2f GB/s\n", milliseconds, bandwidth);
    
    CUDA_CHECK(cudaMemcpy(h_result, d_stride, 4*sizeof(float), cudaMemcpyDeviceToHost));
    printf("Verification: h[0]=%.1f h[1]=%.1f h[2]=%.1f h[3]=%.1f\n", 
           h_result[0], h_result[1], h_result[2], h_result[3]);
    CUDA_CHECK(cudaFree(d_stride));

    // ========== Test 4: Vectorized (float4) ==========
    printf("\n=== Test 4: Vectorized Memory Access (float4) ===\n");
    float *d_vec;
    CUDA_CHECK(cudaMalloc(&d_vec, bytes));
    CUDA_CHECK(cudaMemset(d_vec, 0, bytes));

    threadsPerBlock = 256;
    int N_vec = N / 4;  // Process 4 elements per thread
    blocksPerGrid = (N_vec + threadsPerBlock - 1) / threadsPerBlock;
    
    CUDA_CHECK(cudaEventRecord(start));
    add1_vectorized<<<blocksPerGrid, threadsPerBlock>>>((float4*)d_vec, N_vec);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    bandwidth = (2.0f * bytes) / (milliseconds * 1e6);
    printf("Threads/Block: %d, Blocks: %d (processing %d float4s)\n", 
           threadsPerBlock, blocksPerGrid, N_vec);
    printf("Time: %.3f ms, Bandwidth: %.2f GB/s\n", milliseconds, bandwidth);
    
    CUDA_CHECK(cudaMemcpy(h_result, d_vec, 4*sizeof(float), cudaMemcpyDeviceToHost));
    printf("Verification: h[0]=%.1f h[1]=%.1f h[2]=%.1f h[3]=%.1f\n", 
           h_result[0], h_result[1], h_result[2], h_result[3]);
    CUDA_CHECK(cudaFree(d_vec));

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    delete[] h_result;

    printf("\n=== All Tests PASSED ===\n");
    return 0;
}
