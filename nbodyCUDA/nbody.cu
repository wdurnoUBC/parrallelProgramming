
#include <time.h>
#include <sys/time.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void loadData( char* fileName , int n , float* x , float* y , float* mass , int* actual ) ;
float getVal( char* str , int start , int subLen ) ; 
__global__ void iter( int n , float* xVel , float* yVel ,  float* x , float* y , float* mass , float G , float delt , float* xOut , float* yOut ) ; 

int main( int argc , char** argv ) 
{
	if( argc < 4 ) 
	{
		printf( "REQUIRED: Please provide (arg1) a file to load, (arg2) number of bodies, and (arg3) the number of iterations\n" ) ; 
		printf( "OPTIONAL: (arg4) time-step size in seconds (default=1000.0)\n" ) ; 
		return( 0 ) ; 
	}
	
	float G = 6.67384 * ((float) pow( 10.0 , -11.0 )) ; // Newton's gravitational constant 
	
	int maxIter = atoi( argv[3] ) ; 
	float delt = 1000.0 ; 
	if( argc > 4 ) 
		delt = atof( argv[4] ) ; 
	
	int n = atoi( argv[2] ) ; 
	int m ; 
	float* x = (float*) malloc( n * sizeof(float) ) ; 
	float* y = (float*) malloc( n * sizeof(float) ) ; 
	float* mass = (float*) malloc( n * sizeof(float) ) ; 
	loadData( argv[1] , n , x , y , mass , &m ) ; 
	
	struct timeval startALL, endALL ;
	gettimeofday( &startALL , NULL ) ; 
	
	////////////////// ALLOCATE MEMORY ON DEVICE PRIOR to COMPUTATION 
	float* d_x ; 
	float* d_y ; 
	float* d_mass ; 
	float* d_xVel ; 
	float* d_yVel ; 
	float* d_xTemp ; 
	float* d_yTemp ; 
	float* zeros = (float*) malloc( n * sizeof(float) ) ; 
	int i ; 
	for( i = 0 ; i < n ; i ++ ) 
		zeros[i] = 0.0 ; 
	
	cudaError_t status = cudaMalloc( &d_x , n * sizeof(float) ) ; 
	if( status == cudaSuccess ) 
		status = cudaMalloc( &d_y , n * sizeof(float) ) ; 
	if( status == cudaSuccess ) 
		status = cudaMalloc( &d_mass , n * sizeof(float) ) ; 
	if( status == cudaSuccess ) 
		status = cudaMalloc( &d_xVel , n * sizeof(float) ) ; 
	if( status == cudaSuccess ) 
		status = cudaMalloc( &d_yVel , n * sizeof(float) ) ; 
	if( status == cudaSuccess ) 
		status = cudaMalloc( &d_xTemp , n * sizeof(float) ) ; 
	if( status == cudaSuccess ) 
		status = cudaMalloc( &d_yTemp , n * sizeof(float) ) ; 
	if( status == cudaSuccess ) 
		status = cudaMemcpy( d_x , x , n * sizeof(float) , cudaMemcpyHostToDevice ) ; 
	if( status == cudaSuccess ) 
		status = cudaMemcpy( d_y , y , n * sizeof(float) , cudaMemcpyHostToDevice ) ; 
	if( status == cudaSuccess ) 
		status = cudaMemcpy( d_mass , mass , n * sizeof(float) , cudaMemcpyHostToDevice ) ; 
	if( status == cudaSuccess ) 
		status = cudaMemcpy( d_xVel , zeros , n * sizeof(float) , cudaMemcpyHostToDevice ) ; 
	if( status == cudaSuccess ) 
		status = cudaMemcpy( d_yVel , zeros , n * sizeof(float) , cudaMemcpyHostToDevice ) ; 
	
	//////////////// ASYNCHRONOUS QUEUE JOBS ON DEVICE 
	
	struct timeval startKERNEL , endKERNEL ; 
	gettimeofday( &startKERNEL , NULL ) ; 
	
	for( i = 0 ; i < maxIter ; i+=2 ) 
	{
		iter<<< m/32+1 , 32 >>>( m , d_xVel , d_yVel , d_x , d_y , d_mass , G , delt , d_xTemp , d_yTemp ) ; 
		iter<<< m/32+1 , 32 >>>( m , d_xVel , d_yVel , d_xTemp , d_yTemp , d_mass , G , delt , d_x , d_y ) ; 
	}
	
	if( status == cudaSuccess ) ;
                 status = cudaDeviceSynchronize() ;
	
	gettimeofday( &endKERNEL , NULL ) ; 
	fprintf( stderr , "GPU kernel time: %ld microseconds\n", ((endKERNEL.tv_sec * 1000000 + endKERNEL.tv_usec)
                  - (startKERNEL.tv_sec * 1000000 + startKERNEL.tv_usec)));
	
	//////////////// COPY MEMORY TO HOST
	
	if( status == cudaSuccess ) 
		status = cudaMemcpy( x , d_x , n * sizeof(float) , cudaMemcpyDeviceToHost ) ; 
	if( status == cudaSuccess ) 
		status = cudaMemcpy( y , d_y , n * sizeof(float) , cudaMemcpyDeviceToHost ) ; 
	
	if( status != cudaSuccess )
                printf( "ERROR: %s\n" , cudaGetErrorString(status) ) ; 
	
	cudaFree( d_x ) ; 
	cudaFree( d_y ) ; 
	cudaFree( d_mass ) ; 
	cudaFree( d_xVel ) ; 
	cudaFree( d_yVel ) ; 
	cudaFree( d_xTemp ) ; 
	cudaFree( d_yTemp ) ; 
	
	gettimeofday( &endALL , NULL ) ; 
	fprintf( stderr , "GPU kernel and comm time: %ld microseconds\n", ((endALL.tv_sec * 1000000 + endALL.tv_usec)
                  - (startALL.tv_sec * 1000000 + startALL.tv_usec)));
	
	cudaDeviceReset() ; 
	
	for( i = 0 ; i < m ; i++ ) 
		printf( "%f\t%f\n" , x[i] , y[i] ) ; 
	
	return( 0 ) ; 
}

__global__ void iter( int n , float* xVel , float* yVel ,  float* x , float* y , float* mass , float G , float delt , float* xOut , float* yOut )
{
	int rank = threadIdx.x + blockIdx.x * blockDim.x ; 
	if( rank >= n ) 
		return ; 
        int i ;
        float r ;
        float xForce = 0.0 ;
        float yForce = 0.0 ;
        for( i = 0 ; i < n ; i++ )
        {
                if( i != rank )
                {
                        // Calculations are done in exponentiated logs to reduce roundoffs
                        r = sqrt( (x[i] - x[rank])*(x[i] - x[rank]) + (y[i] - y[rank])*(y[i] - y[rank]) ) ;
                        if( x[i] > x[rank] )
                                xForce = xForce + exp( log(G) + log(mass[i]) + log(mass[rank]) + log( x[i] - x[rank] ) - 3.0*log( r ) ) ;
                        if( x[i] < x[rank] )
                                xForce = xForce - exp( log(G) + log(mass[i]) + log(mass[rank]) + log( x[rank] - x[i] ) - 3.0*log( r ) ) ;
                        // case: x[i] == x[rank] : do nothing
                        if( y[i] > y[rank] )
                                yForce = yForce + exp( log(G) + log(mass[i]) + log(mass[rank]) + log( y[i] - y[rank] ) - 3.0*log( r ) ) ;
                        if( y[i] < y[rank] )
                                yForce = yForce - exp( log(G) + log(mass[i]) + log(mass[rank]) + log( y[rank] - y[i] ) - 3.0*log( r ) ) ;
                        // case: y[i] == y[rank] : do nothing
                }
        }
        xVel[rank] = xVel[rank] + xForce * delt / mass[rank] ;
        yVel[rank] = yVel[rank] + yForce * delt / mass[rank] ;
        xOut[rank] = x[rank] + xVel[rank] * delt ;
        yOut[rank] = y[rank] + yVel[rank] * delt ;
}

void loadData( char* fileName , int n , float* x , float* y , float* mass , int* actual )
{
        char temp[1000] ;
        FILE *file ;
        file = fopen( fileName , "r" ) ;
        if( file == NULL )
        {
                printf( "File failed to open!\n" ) ;
                return ;
        }

        *actual = 0 ;
        int delim1 , delim2 , delim3 ; // ends of delimeters
        int len , i , j , flag ;
        // char temp1[1000] ;
        for( j = 0 ; fgets( temp , 1000 , file ) != NULL && j < n ; j++ )
        {
                len = strlen( temp ) ;
                delim1 = -1 ;
                for( i = 0 ; i < len && delim1 < 0 ; i++ )
                {
                        if( temp[i] != ' ' )
                                delim1 = i ;
                }
                delim2 = -1 ;
                flag = -1 ;
                for( i = delim1 + 1 ; i < len && delim2 < 0 ; i++ )
                {
                        if( temp[i] == ' ' )
                                flag = 1 ;
                        if( temp[i] != ' ' && flag > 0 )
                                delim2 = i ;
                }
                delim3 = -1 ;
                flag = -1 ;
                for( i = delim2 + 1 ; i < len && delim3 < 0 ; i++ )
                {
                        if( temp[i] == ' ' )
                                flag = 1 ;
                        if( temp[i] != ' ' && flag > 0 )
                                delim3 = i ;
                }
                if( delim1 < 0 || delim2 < 0 || delim3 < 0 )
                {
                        printf( "Input data formatting error\n" ) ;
                        return ;
                }

                x[j] = getVal( temp , delim1 , delim2 - delim1 ) ;
                y[j] = getVal( temp , delim2 , delim3 - delim1 ) ;
                mass[j] = getVal( temp , delim3 , -1 ) ;
		
		*actual = *actual + 1 ;
        }
        fclose( file ) ;
}

float getVal( char* str , int start , int subLen )
{
        int len = strlen( str ) ;
        if( subLen < 0 )
                subLen = len - start + 1 ;
        else
                subLen = subLen + 1 ;
        char temp[subLen] ;
        temp[subLen - 1] = '\0' ;
        int i ;
        for( i = 0 ; i < subLen - 1 ; i++ )
        {
                temp[i] = str[i+start] ;
        }
        return( atof( temp ) ) ;
}

