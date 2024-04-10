#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define imin(a,b) (a<b?a:b)

const int N = 32*1204;
const int Threads_per_block = 256;
const int Blocks_per_grid = imin (32, (N +Threads_per_block - 1) / Threads_per_block);

__global__ void dot(float *a, float *b, float *c)
{
	__shared__ float cache[Threads_per_block];
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	int cacheIndex = threadIdx.x;
	float temp = 0;

	while (tid < N){
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}

	cache[cacheIndex] = temp;

	__syncthreads();

	//reduction operation in log2 time
    	int i = blockDim.x /2;
	while (i != 0){
		if (cacheIndex < i){
			cache[cacheIndex] += cache[cacheIndex + i];
		}
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0){
		c[blockIdx.x] = cache[0];
	}
}

int main(void){
	float *a, *b, c, *partial_c;
	float *d_a, *d_b, *d_c; 
	

	a = (float *) malloc(sizeof(float)*N);	
	b = (float *) malloc(sizeof(float)*N);
	partial_c = (float *) malloc(sizeof(float)*Blocks_per_grid);

	c = 0;
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaMalloc((void**)&d_a, sizeof(float)*N);
	cudaMalloc((void**)&d_b, sizeof(float)*N);
	cudaMalloc((void**)&d_c, sizeof(float)*Blocks_per_grid);

	for (int i = 0; i < N; i++){
		a[i] = i;
		b[i] = i * 2;
	}

	cudaMemcpy(d_a, a, sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(float)*N, cudaMemcpyHostToDevice);

	dot <<< Blocks_per_grid, Threads_per_block >>> (d_a,d_b,d_c);

	cudaMemcpy(partial_c, d_c, sizeof(float)*Blocks_per_grid, cudaMemcpyDeviceToHost);
	c = 0;
	for (int i = 0; i < Blocks_per_grid; i++){
		c += partial_c[i];
		//printf("partial_c %f", partial_c[i]);
	}
	
	#define sum_squares(x)  (x*(x+1)*(2*x+1)/6)
	printf("Does GPU %.6g = %.6g? \n", c, 2 * sum_squares((float)(N - 1)));

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);


	free(a);
	free(b);
	free(partial_c);
	return 0;
}