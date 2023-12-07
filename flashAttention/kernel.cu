#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

// attention_kernel.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <algorithm>

using namespace std;

__global__ void flashAttention(float* Q, float* K, float* V, float* O, int d, int N)
{
	int x = threadIdx.x;

	int i = blockIdx.x;

	__shared__ float Kj[32 * 64];
	__shared__ float Vj[32 * 64];
	__shared__ float Qi[32 * 64];
	__shared__ float Oi[32 * 64];
	__shared__ float temp[32 * 64];

	__shared__ float Sij[32 * 32];

	__shared__ float mi[4 * 32];
	__shared__ float li[4 * 32];

	__shared__ float lij[32];
	__shared__ float mij[32];

	__shared__ float li_new[4 * 32];
	__shared__ float mi_new[4 * 32];

	int T_c = 4;
	int T_r = 4;

	float l_new = 0;

	for (int q = x; q < 4 * 32; q += 32) {
		mi[q] = -INFINITY;
		mi_new[q] = -INFINITY;
		li[q] = 0;
		li_new[q] = 0;
	}

	for (int j = 0; j < T_c; j++)
	{
		for (int q = x; q < 32 * 64; q += 512)
		{
			Kj[q] = K[q + (j * (32 * 64))];
			Vj[q] = V[q + (j * (32 * 64))];

			Oi[q] = O[q + (i * (32 * 64))];
			Qi[q] = Q[q + (i * (32 * 64))];

			if (q < 32)
			{
				lij[q] = 0;
				mij[q] = -INFINITY;
			}

		}
		__syncthreads();

		for (int q = x; q < 32 * 32; q += 512)
		{
			Sij[q] = 0;

			int row = q / 32;
			int col = q % 32;

			for (int k = 0; k < 64; k++)
			{
				Sij[q] += Qi[row * 64 + k] * Kj[col * 64 + k];

			}

		}

		__syncthreads();

		if (x < 32)
		{
			for (int q = 0; q < 32; q++)
			{
				mij[x] = max(mij[x], Sij[x * 32 + q]);
			}
		}

		__syncthreads();

		for (int q = x; q < 32 * 32; q += 512)
		{
			int row = q / 32;
			int col = q % 32;



			Sij[q] = exp(Sij[q] - mij[row]);

			atomicAdd(&lij[row], Sij[q]);
		}

		for (int q = x; q < 32; q += 512)
		{
			mi_new[(i * 32) + q] = max(mi[(i * 32) + q], mij[q]);
			li_new[(i * 32) + q] = exp(mi[(i * 32) + q] - mi_new[(i * 32) + q]) * li[(i * 32) + q] + exp(mij[q] - mi_new[(i * 32) + q]) * lij[q];
		}

		for (int q = x; q < 32 * 64; q += 512)
		{
			int row = q / 64;
			int col = q % 64;

			temp[q] = 0;

			for (int k = 0; k < 32; k++)
			{
				temp[q] += Sij[row * 32 + k] * Vj[k * 64 + col];
			}

			/*if (temp[q] != 32.0f)
			{
				printf("temp[%d] = %f\n", q, temp[q]);
			}*/
		}



		__syncthreads();

		for (int q = x; q < 32 * 64; q += 512)
		{
			int row = q / 64;

			Oi[q] = (li[(i * 32) + row] * exp(mi[(i * 32) + row] - mi_new[(i * 32) + row]) * Oi[q] + exp(mij[row] - mi_new[(i * 32) + row]) * temp[q]) / li_new[(i * 32) + row];
			O[i * (32 * 64) + q] = Oi[q];
		}

		if (x < 32)
		{
			mi[(i * 32) + x] = mi_new[(i * 32) + x];
			li[(i * 32) + x] = li_new[(i * 32) + x];
		}

		__syncthreads();
	}
}


void flashAttentionLauncher(float* Q, float* K, float* V, float* O, int N, int d)
{

	float* l = new float[32];
	float* m = new float[32];

	for (int i = 0; i < 32; i++)
	{
		l[i] = 0;
		m[i] = -INFINITY;
		for (int j = 0; j < d; j++)
		{
			O[i * d + j] = 0;
		}
	}

	float* d_Q;
	float* d_K;
	float* d_V;
	float* d_O;
	float* d_l;
	float* d_m;

	cudaMalloc((void**)&d_Q, N * d * sizeof(float));
	cudaMalloc((void**)&d_K, N * d * sizeof(float));
	cudaMalloc((void**)&d_V, N * d * sizeof(float));
	cudaMalloc((void**)&d_O, N * d * sizeof(float));
	cudaMalloc((void**)&d_l, 32 * sizeof(float));
	cudaMalloc((void**)&d_m, 32 * sizeof(float));

	cudaMemcpy(d_Q, Q, N * d * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_K, K, N * d * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V, V, N * d * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_O, O, N * d * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_l, l, 32 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_m, m, 32 * sizeof(float), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(512);
	dim3 numBlocks(4);

	flashAttention << <numBlocks, threadsPerBlock >> > (d_Q, d_K, d_V, d_O, d, N);

	cudaMemcpy(O, d_O, N * d * sizeof(float), cudaMemcpyDeviceToHost);
	//print output
	/*for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < d; j++)
		{
			printf("%f ", O[i * d + j]);
		}
		printf("\n");
	}*/
}


int main()
{

	int d = 64;
	int N = 128;
	float scale = 1.0f;

	//test scaled_dot_product_attention
	float* q = (float*)malloc(static_cast<unsigned long long>(N) * d * sizeof(float));
	float* k = (float*)malloc(static_cast<unsigned long long>(N) * d * sizeof(float));
	float* v = (float*)malloc(static_cast<unsigned long long>(N) * d * sizeof(float));
	float* output = (float*)malloc(static_cast<unsigned long long>(N) * d * sizeof(float));

	if (q == NULL || k == NULL || v == NULL || output == NULL) {
		printf("Memory allocation failed");

		// Free any allocations that were successful
		free(q);
		free(k);
		free(v);
		free(output);

		return -1;
	}

	//initializeRandomMatrix(q, 128, 64);
	//initializeRandomMatrix(k, 128, 64);
	//initializeRandomMatrix(v, 128, 64);

	for (int i = 0; i < N * d; i++)
	{
		q[i] = 1.0f;
		k[i] = 2.0f;
		v[i] = 12.0f;
		output[i] = 0.0f;
	};

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	for (int i = 0; i < 100; i++) {
		flashAttentionLauncher(q, k, v, output, N, d);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float milliseconds = 0;

	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("Time for kernel execution: %f ms", milliseconds / 100);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);



	return 0;
}