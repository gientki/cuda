#include<iostream>
#include<cuda_runtime.h>

using namespace std;


__global__ void sruchom(int* a, int* c, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int temp[64];

	if (idx < size)
		temp[threadIdx.x] = a[idx];

	__syncthreads();  

	if (idx > 0 && idx < size - 1) {
		c[idx] = (temp[threadIdx.x - 1] + temp[threadIdx.x] + temp[threadIdx.x + 1]) / 3.0f;
	}
	else if (idx < size) {
		c[idx] = a[idx];
	}
}
int main()
{
	const int size = 64;
	const int blocks = 1;

	int h_a[size];
	int h_b[size];
	int h_c[size];
	
	for (int i = 0; i < size; i++)
	{
		h_a[i] = i;
		h_b[i] = i;
	}
	
	int* d_a, * d_b, * d_c;
	
	
	cudaMalloc((void**)&d_a, size * sizeof(int));
	cudaMalloc((void**)&d_b, size * sizeof(int));
	cudaMalloc((void**)&d_c, size * sizeof(int));
	/*cout << d_a << endl;*/
	cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice);
	
	/*addKernel<<<blocks, size/blocks>>>(d_a, d_b, d_c, size);*/
	sruchom<<<blocks, size/blocks>>>(d_a, d_c, size);

	cudaMemcpy(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < size; i++)
		cout<<i<<':'<< h_c[i] << endl;

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
		return 0;
}