//Matrix linearization

#include "cuda_runtime.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define N 4096  //size of problem will be N*N matrix
#define THREADS_PER_BLOCK 512   //size of block

__global__ void add(int *a, int *b, int *c, int max){
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	int id = index;
	while (id < max){
		c[id] = a[id] + b[id];
		// for problems larger than number of threads increase
		// the index is increased by the total number of threads
		id =  id + blockDim.x* gridDim.x;    
	}
}

void fill_mat(int *mat ){
	int c = 0;
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			mat[j + i*N] = 1;
		}
	}
}


void print_mat(int *mat){
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			printf("%i \t", mat[j+i*N]);
		}
		printf("\n");
	}
	printf("\n");
}

int main(){
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	int tam = N*N*sizeof(int);

	cudaMalloc((void**)&d_a, tam);
	cudaMalloc((void**)&d_b, tam);
	cudaMalloc((void**)&d_c, tam);


	a = (int*)malloc(tam);
	b = (int*)malloc(tam);
	c = (int*)malloc(tam);

	fill_mat(a);
	fill_mat(b);
	
	//print_mat(a);
	//print_mat(b);

	cudaMemcpy(d_a, a, tam, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, tam, cudaMemcpyHostToDevice);


	add << <N*N / THREADS_PER_BLOCK,  THREADS_PER_BLOCK >> >(d_a, d_b, d_c, N*N);

	cudaMemcpy(c, d_c, tam, cudaMemcpyDeviceToHost);
	
	//print_mat(c);
	

	long int i, acum = 0;

	for(i = 0; i< N*N; i++){
		acum = acum + c[i];
	}

	printf("\n addition of values %i \n", acum);

	free(a);
	free(b);
	free(c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}