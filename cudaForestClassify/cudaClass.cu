
#include <time.h>
#include <sys/time.h>

#include "types.h"

#include <stdio.h>

int maxLab( int thisNode , cuNode* cuNodes )
{
	int out = cuNodes[thisNode].leftAttr ; 
	if( out < cuNodes[thisNode].rightAttr ) 
		out = cuNodes[thisNode].rightAttr ; 
	int temp ;
	if( cuNodes[thisNode].left > -1 )
		temp = maxLab( cuNodes[thisNode].left , cuNodes ) ; 
	if( out < temp ) 
		out = temp ; 
	if( cuNodes[thisNode].right > -1 ) 
		temp = maxLab( cuNodes[thisNode].right , cuNodes ) ; 
	if( out < temp ) 
		out = temp ; 
	return( out ) ; 
}

__global__ void makeContact( cuNode* cuNodes , int* starts , int* gpuTempSpace , int labelMax , int trees , float* datum )
{
        int idx = threadIdx.x + blockIdx.x * blockDim.x ; 
        if( idx >= trees )
                return ; 
	
	//////// PART 1 : EVALUATE TREES 
	
	int i , j , k ; 
	for( i = 0 ; i < labelMax + 1 ; i++ ) 
		gpuTempSpace[ (labelMax+1)*idx + i ] = 0 ; // set all initial votes to zero 
	
	i = starts[idx] ; 
	while( i >= 0 ) // classify with this one tree 
	{
		j = i ; // store previous step 
		( datum[ cuNodes[i].dim ] < cuNodes[i].rule ) ? ( i = cuNodes[i].left ) : ( i = cuNodes[i].right ) ; // This avoid alternate execution paths on GPU 
	}
	
	( datum[ cuNodes[j].dim ] < cuNodes[j].rule ) ? gpuTempSpace[ (labelMax+1)*idx + cuNodes[j].leftAttr] = 1 : gpuTempSpace[ (labelMax+1)*idx + cuNodes[j].rightAttr] = 1 ; 
	
	__syncthreads() ; 
	
	//////// PART 2 : LOG-TIME SUM VOTES 
	
	while( trees > 1 && idx < trees ) 
	{
		( trees % 2 == 0 ) ? ( j = 1 ) : ( j = 0 ) ; // j is the even flag 
		( j > 0 ) ? ( trees = trees/2 ) : ( trees = (trees-1)/2+1 ) ; 
		
		if( idx >= trees ) // Drop useless threads 
			return ; 
		
		__syncthreads() ; 
		
		for( i = 0 ; i < labelMax+1 ; i++ ) 
		{
			if( j > 0 ) // Paths do NOT bifurcate here, because j is the same for all threads 
			{
				k = gpuTempSpace[ (labelMax+1)*(2*idx) + i ] + gpuTempSpace[ (labelMax+1)*(2*idx+1) + i ] ; 
			}
			else // odds work 
			{
				( idx == trees - 1 ) ? ( k = gpuTempSpace[ (labelMax+1)*(2*idx) + i ] ) : ( k = gpuTempSpace[ (labelMax+1)*(2*idx) + i ] + gpuTempSpace[ (labelMax+1)*(2*idx+1) + i ] ) ; 
			}
			
			__syncthreads() ; 
			
			gpuTempSpace[ (labelMax+1)*idx + i ] = k ; 
			
			__syncthreads() ;  
		}
	}
	
}

void loadDataToGPU( cuNode* cuNodes , int cuNodesN , int* starts , int trees , float* datum , int dims )
{
	cudaDeviceReset() ; 
	
	int maxLabel = 0 ; 
	int i ; 
	for( i = 0 ; i < cuNodesN ; i++ )
	{
		if( maxLabel < cuNodes[i].leftAttr ) 
			maxLabel = cuNodes[i].leftAttr ; 
		if( maxLabel < cuNodes[i].rightAttr ) 
			maxLabel = cuNodes[i].rightAttr ; 
	}
	printf( "DEBUG: maxlabel: %i\n" , maxLabel ) ; 
	cuNode* d_cuNodes ; 
	int* d_tempSpace ; 
	int* d_starts ; 
	cudaError_t status = cudaMalloc( &d_cuNodes , cuNodesN * sizeof(cuNode) ) ;  
	if( status == cudaSuccess ) 
		status = cudaMalloc( &d_tempSpace , (maxLabel + 1) * trees * sizeof(int) ) ; 
	if( status == cudaSuccess ) 
		status = cudaMalloc( &d_starts , trees * sizeof(int) ) ; 
	if( status == cudaSuccess ) 
		status = cudaMemcpy( d_cuNodes , cuNodes , cuNodesN * sizeof(cuNode) , cudaMemcpyHostToDevice ) ; 
	if( status == cudaSuccess ) 
		status = cudaMemcpy( d_starts , starts , trees * sizeof(int) , cudaMemcpyHostToDevice ) ; 
	
	struct timeval startALL, endALL , startAttr , endAttr ;
	
	gettimeofday( &startALL , NULL ) ; 
	
	// load datum to GPU for classification 
	float* d_datum ; 
	if( status == cudaSuccess ) 
		status = cudaMalloc( &d_datum , dims * sizeof(float) ) ; 
	if( status == cudaSuccess ) 
		status = cudaMemcpy( d_datum , datum , dims * sizeof(float) , cudaMemcpyHostToDevice ) ; 
	
	gettimeofday( &startAttr , NULL ) ;
	
	// classify datum 
	if( status == cudaSuccess ) 
	{
		makeContact<<< 1 , trees >>>( d_cuNodes , d_starts , d_tempSpace , maxLabel , trees , d_datum ) ; // Trees cannot yet be larger than 1024 !!! 
		status = cudaDeviceSynchronize() ; 
	}
	
	gettimeofday( &endAttr , NULL ) ;
	
	// load result from GPU 
	int* out = (int*) malloc( (maxLabel+1) * sizeof(int) ) ; 
	if( status == cudaSuccess ) 
		status = cudaMemcpy( out , d_tempSpace , (maxLabel+1) * sizeof(int) , cudaMemcpyDeviceToHost ) ; 
	
	gettimeofday( &endALL , NULL ) ;
	
	int label = 0 ; 
	int votes = 0 ; 
	
	for( i = 0 ; i < maxLabel+1 ; i++ ) 
	{
		if( votes < out[i] ) 
		{
			label = i ; 
			votes = out[i] ; 
		}
	}
	
	printf("CUDA all: %ld\n", ((endALL.tv_sec * 1000000 + endALL.tv_usec)
		  - (startALL.tv_sec * 1000000 + startALL.tv_usec)));
	
	printf("CUDA attr: %ld\n", ((endAttr.tv_sec * 1000000 + endAttr.tv_usec)
		  - (startAttr.tv_sec * 1000000 + startAttr.tv_usec)));
	
	printf( "CUDA attributes datum as label %i\n" , label ) ; 
	
	for( i = 0 ; i < maxLabel+1 ; i++ )
		printf( "%i: %i\n" , i , out[i] ) ; 
	
	if( status != cudaSuccess ) 
		printf( "ERROR: %s\n" , cudaGetErrorString(status) ) ; 
	
	
	cudaFree( d_cuNodes ) ; 
	cudaFree( d_starts ) ; 
	cudaFree( d_tempSpace ) ; 
	free( out ) ; 
	cudaDeviceReset() ; 
}











