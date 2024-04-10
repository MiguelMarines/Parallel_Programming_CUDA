%%cu

// Laboratory 02 - Parallelism Cuda
// Miguel Marines

// Libraries
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


// Function for the matrix multiplication
__global__ void MatrixMultiplicationGPU(float *d_matrixA, float *d_matrixB, float *d_matrixR, int N)
{
	// Compute row and colum index for each thread
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

  	d_matrixR[row * N + col] = 0;
  	
	// Loop to iterate over row and down column
	for (int i = 0; i < N; i++)
	{
    	// Accumulate results for a single element
    	d_matrixR[row * N + col] += d_matrixA[row * N + i] * d_matrixB[i * N + col];
  	}
}


// Main
int main()
{
	// Square matrix dimension
	int N = 10;
		
	// Threads per block
	int threads_per_block = N;
	
	// Number of Blocks
	int blocks = (N + threads_per_block - 1) / threads_per_block;
	
	// Matrices Host - CPU
	float h_matrixA[N][N], h_matrixB[N][N], h_matrixR[N][N];

	// Matrices Device - GPU
	float *d_matrixA, *d_matrixB, *d_matrixR;

	// Matrix Size
	int size = sizeof(float) * N * N;

	// Memory reserve in the GPU
	cudaMalloc((void**)&d_matrixA, size);
	cudaMalloc((void**)&d_matrixB, size);
	cudaMalloc((void**)&d_matrixR, size);
	
	// Fill matrix A
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			h_matrixA[i][j] = rand() % 100;
		}
	}
	
	// Fill matrix B
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			h_matrixB[i][j] = rand() % 100;
		}
	}

	// Print matrix A
	printf("Matrix A:\n");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%.2f  ", h_matrixA[i][j]);
		}
		printf("\n");
	}
	printf("\n");
	
	// Print matrix B
	printf("Matrix B:\n");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%.2f  ", h_matrixB[i][j]);
		}
		printf("\n");
	}
	printf("\n");


	// Copy values from the CPU to the GPU
	cudaMemcpy(d_matrixA, h_matrixA, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, size, cudaMemcpyHostToDevice);

	// Initialize objects

	// Number of blocks in the matrix
	dim3 Blocks(blocks, blocks);
	
	// Number of threads per block
	dim3 Threads(threads_per_block, threads_per_block);
	
	// Kernel call
	MatrixMultiplicationGPU <<<Blocks, Threads>> >(d_matrixA, d_matrixB, d_matrixR, N);

	// Copy values from the GPU to the CPU
	cudaMemcpy(h_matrixR, d_matrixR, size, cudaMemcpyDeviceToHost);

	// Print result matrix
	printf("Matrix R:\n");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%.2f  ", h_matrixR[i][j]);
		}
		printf("\n");
	}
	
	// Free memory
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixR);

	// Terminate program
	return 0;
}