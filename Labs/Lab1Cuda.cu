%%cu

// Laboratory 01 Parallelism Cuda
// Miguel Marines

// Libraries
# include "cuda_runtime.h"
# include <stdio.h>

// Constants
# define NUMBER_RECTANGLES 100000000
# define THREADS_PER_BLOCK 256

// Function to calculate PI
__global__ void calculatePIgpu(double *sum, long number_rectangles)
{
		// Variables.
		double mid, height;
	
		// Calculation of the width.
		double width = 1.0 / (long) number_rectangles;
	
		// Calculation of the index.
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
	
		if(index < number_rectangles)
  	{
				// Algorithm to calculate the sum.
				mid = (index + 0.5) * width;
				height = 4.0 / (1.0 + (mid * mid));
				sum[index]= height;
		}
 
}


int main()
{
		// Variables for the CPU (Host).
		double *sum;
		double width, area, addition = 0.0;

		// Variables for the GPU (Device).
		double *d_sum;
	
		// Size of the memory
		double size = NUMBER_RECTANGLES * sizeof(long);

		// Reserve memory in the GPU.
		cudaMalloc((void**) &d_sum, size);
		sum = (double*)malloc(size);

		// Kernel execution.
		calculatePIgpu<<<NUMBER_RECTANGLES/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_sum, NUMBER_RECTANGLES);

		// Copy value from the GPU(Device) to the CPU(Host).
		cudaMemcpy(sum, d_sum, size, cudaMemcpyDeviceToHost);

		// Addition of the values obtained from the calculations in the GPU.
		for(int i = 0; i < NUMBER_RECTANGLES; i++)
 		{
			addition = addition + sum[i];
		}
	
		// Calculate width.
		width = 1.0 / (long) NUMBER_RECTANGLES;
		
		// Calculate area (PI).
		area = width * addition;

		// Print the result of the calculation of PI.
		printf("pi = %f", area);
	
		// Free memory from CPU (Host).
		free(sum);
		
		// Free memory from GPU (Device).
		cudaFree(d_sum);
		
		// Teminate execution of the program.
		return 0;
}