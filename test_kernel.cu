#include <cuda.h>

#include <nvToolsExt.h>

#include <cuda_profiler_api.h>

#include <stdio.h>

#include <pthread.h>

//#include "utils.h"



#define BLOCKS 80

#define THREADS 512

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }   \
} 


__global__ void bar() {

	__shared__ int a[THREADS];

	int tid = threadIdx.x;

	for (int i = 0; i < 1000000; i++) {

		a[tid] += tid + i;

	}

//	if (tid==10) printf("%d\n", a[tid]);

}



int foo() {

	int sum = 0;

	for(int i = 0; i <  1000000; i++) {

		sum += i;

	}

	return sum;

}



int main(void) {

	cudaProfilerStart();


	nvtxNameOsThread(pthread_self(), "MAIN");



	nvtxRangePush("Calling foo");

	printf("%d\n", foo());

	nvtxRangePop();



	nvtxRangePush("Calling bar1");

	bar<<<BLOCKS, THREADS>>>();

	nvtxRangePop();

        cudaCheckError();


	nvtxRangePush("Calling bar2");

	bar<<<BLOCKS, THREADS>>>();

	nvtxRangePop();

        cudaCheckError();



	nvtxRangePush("Calling bar3");

	bar<<<BLOCKS, THREADS>>>();

	nvtxRangePop();

        cudaCheckError();



	cudaDeviceSynchronize();



	cudaProfilerStop();

	return 0;

}
