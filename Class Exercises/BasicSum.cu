%%cu

#include "cuda_runtime.h"
#include <stdio.h>


// global function is the main of the gpu
__global__ void suma(int *a, int *b, int *c){
	*c = *a + *b;
}


int main(){
	//variables for the cpu (host)
	int a, b, c;
	
	//variables for the gpu (device)
	int *d_a, *d_b, *d_c;
	
	int size = sizeof(int);
	
	//reserve memory in the GPU
	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);
	
	//test values
    a = 10;
    b = 20;

	//copy the values from the cpu(host) to the gpu(device)
	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
	
	//kernel call with 1 thread and 1 block
	suma << <1, 1 >> >(d_a, d_b, d_c);

	//copy the values from the gpu(device) to the cpu(host)
	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
	
	printf("result is  %d\n", c);

    //free memory from the cuda pointers
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;

}