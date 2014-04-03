
#include <stdio.h>
#include <stdlib.h>

#include <time.h>
#include <sys/time.h>

__global__ void flops( float* floats , int n , int m )
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x ; 
	if( idx >= m ) 
		return ; 
	float temp  = 3.14159 * idx ; 
	int i ; 
	for( i = 0 ; i < n ; i++ )
		temp = temp + temp/2.0 ; 
	floats[idx] = temp ; 
}

int main( int argc , char** argv ) 
{
	cudaDeviceReset() ; 
	
	struct timeval start, end ; 
	
	int n = 10 ; 
	if( argc > 1 ) 
		n = atoi(argv[1]) ; 
	else
		printf( "Optional arguments: (arg1) number of floats to add and divide, (arg2) number of threads \n" ) ; 
	int m = 100 ; 
	if( argc > 2 ) 
		m = atoi(argv[2]) ; 
	
	int blocks = m/32 + 1 ; 
	
	float* d_floats ;
	cudaError_t status = cudaMalloc( &d_floats , m * sizeof(float) ) ;  
	float* h_floats = (float*) malloc( m * sizeof(float) ) ; 
	
	gettimeofday(&start, NULL) ;
	if( status == cudaSuccess ) 
	{
		flops<<< blocks , 32 >>>( d_floats , n , m ) ; 
		status = cudaDeviceSynchronize() ; 
	}
	
	gettimeofday(&end, NULL) ;
	
	  printf("%ld microseconds on GPU\n", ((end.tv_sec * 1000000 + end.tv_usec)
		  - (start.tv_sec * 1000000 + start.tv_usec)));  
	
	if( status == cudaSuccess ) 
		status = cudaMemcpy( h_floats , d_floats , m * sizeof(float) , cudaMemcpyDeviceToHost ) ; 
	
	if( status != cudaSuccess ) 
		printf( "ERROR: %s\n" , cudaGetErrorString(status) ) ; 
	
	float out = 0.0 ; 
	
	gettimeofday(&start, NULL) ; 
	
	int i , j ;
	for( i = 0 ; i < m ; i++ )
	{
		float temp = 3.14159 * i ; 
		for( j = 0 ; j < n ; j++ )
			out = temp + temp / 2.0 ; 
	}
	
	gettimeofday(&end, NULL) ; 
	
	printf("%ld microseconds on CPU\n", ((end.tv_sec * 1000000 + end.tv_usec)
                  - (start.tv_sec * 1000000 + start.tv_usec)));
	
	out = 0.0 ; 
	for( i = 0 ; i < m ; i++ )
		out += h_floats[i] ; 
	printf( "Total val: %f \n" , out ) ; 
	
	free(h_floats) ; 
	cudaFree(d_floats) ; 
	
	cudaDeviceReset() ; 
}















