#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

__global__ void stress_kernel(float *a) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float x = idx;
  for (int i = 0; i < 1000000; i++) x = sinf(x) + cosf(x);
  a[idx] = x;
}

int main() {
  const int N = 1 << 20; // 1M threads
  float *d_a;
  cudaMalloc(&d_a, N * sizeof(float));

  for (int i = 0; i < 10; ++i) {
    stress_kernel<<<N / 256, 256>>>(d_a);
    cudaDeviceSynchronize();
  }

  cudaFree(d_a);
  return 0;
}

