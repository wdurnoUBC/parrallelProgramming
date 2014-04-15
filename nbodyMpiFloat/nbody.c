
#include <time.h>
#include <sys/time.h>

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

float getVal( char* , int , int ) ;
void loadData( char* fileName , int n , float* x , float* y , float* mass , int* actual ) ;
void master( char* , int , float , float ) ;
void slave( int , float , float ) ;

void sendInitData( int nextGuy , int n , float* x , float* y , float* mass ) ;
void recvInitData( int from , int* n , float** x , float** y , float** mass ) ;
void updateLocalVars( int , int , float* , float* ,  float* , float* , float* , float , float ) ; 
void sendUpdate( int nextGuy , float* x , float* y , int n , int i ) ; 
void recvUpdate( int from , float* x , float* y , int n , int i ) ; 

int main( int argc , char** argv )
{
	MPI_Init(&argc, &argv);
	
	int rank ;
	MPI_Comm_rank( MPI_COMM_WORLD , &rank ) ;
	
	if( argc < 3 && rank == 0 )
	{
		printf( "Please provide the name (arg1) of a space-delimited file storing a real-valued matrix with a positive number of rows, 3 columns, and the last column positive.\n" ) ;
		printf( "Please also pass a positive integer (arg2) indicating the number of simulation cycles\n" ) ;
		printf( "Optionally, provide (arg3) the time step size, default is 1000s\n" ) ;
		return( 1 ) ;
	}
	
	float G = 6.67384 * pow( 10.0 , -11.0 ) ;
	float delt = 1000 ; 
	if( argc > 3 )
		delt = atof( argv[3] ) ; 
	
	int iterSize = atoi( argv[2] ) ;
	
	struct timeval start , end ; 
	if( rank == 0 ) 
		gettimeofday( &start , NULL ) ; 
	
	if( rank == 0 )
		master( argv[1] , iterSize , G , delt ) ;
	else
		slave( iterSize , G , delt ) ;
	
	if( rank == 0 )
	{
		gettimeofday( &end , NULL ) ; 
		fprintf( stderr , "CPU time: %ld microseconds\n", ((end.tv_sec * 1000000 + end.tv_usec)
                  - (start.tv_sec * 1000000 + start.tv_usec)));
	}
	
	MPI_Finalize() ;
	return( 0 ) ;
}

void master( char* fileName , int N , float G , float delt )
{
	// INITIALIZATION
	int n , m ;
	MPI_Comm_size (MPI_COMM_WORLD, &n) ;
	float x[n] ;
	float y[n] ;
	float mass[n] ;
	int BUGS = N ;
// printf( "DEBUG: the iterSize is %i, 1\n" , N ) ;
	loadData( fileName , n , x , y , mass , &m ) ;
	N = BUGS ;
// printf( "DEBUG: the iterSize is %i, 2\n" , N ) ;
	if( m < n )
	{
		printf( "WARNING: fewer data than workers, operating on %i of %i workers.\n" , m , n ) ;
		n = m ;
	}
	
	if( n < 2 )
	{
		printf( "ERROR: not enough points for an nbody simulation\n" ) ;
		return ;
	}
	sendInitData( 1 , n , x , y , mass ) ;
	
	float xVel = 0.0 ;	// Assumes initial veolocity is zero
	float yVel = 0.0 ;
	
	// SIMULATION
	
	int i ;
	float r ;
	for( i = 0 ; i < N ; i++ )
	{
		// Update local variables
		updateLocalVars( 0 , n , &xVel , &yVel , x , y , mass , G , delt ) ;
		
		// Send updated local information
		sendUpdate( 1 , x , y , n , i ) ;
		
		// Receive updated positions
		recvUpdate( n-1 , x , y , n , i ) ;
	}
	
	// TERMINATION
	
	// write output file
	for( i = 0 ; i < n ; i++ )
		printf( "%f %f\n" , x[i] , y[i] ) ; 
}

void slave( int N , float G , float delt )
{
	// INITIALIZATION
	int rank ;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank) ;
	
	int n ;
	float* x ;
	float* y ;
	float* mass ;
	recvInitData( rank-1 , &n , &x , &y , &mass ) ;
	if( rank != n-1 )
		sendInitData( rank+1 , n , x , y , mass ) ;
	
	// SIMULATION
	float xVel = 0.0 ;
	float yVel = 0.0 ;
	
	int i ;
	float xTemp , yTemp ;
	for( i = 0 ; i < N ; i++ )
	{
		// update local variables
		updateLocalVars( rank , n , &xVel , &yVel , x , y , mass , G , delt ) ; 
		
		// receive updated positions
		xTemp = x[rank] ;
		yTemp = y[rank] ;
		recvUpdate( rank-1 , x , y , n , i ) ;
		x[rank] = xTemp ;
		y[rank] = yTemp ;
		
		// send updated positions
		if( rank < n-1 )
			sendUpdate( rank+1 , x , y , n , i ) ;
		else
			sendUpdate( 0 , x , y , n , i ) ;
	}
	
	// TERMINATION
	// all work is already complete!
}

void updateLocalVars( int rank , int n , float* xVel , float* yVel ,  float* x , float* y , float* mass , float G , float delt )
{
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
	*xVel = (*xVel) + xForce * delt / mass[rank] ;
	*yVel = (*yVel) + yForce * delt / mass[rank] ;
	x[rank] = x[rank] + (*xVel) * delt ;
	y[rank] = y[rank] + (*yVel) * delt ; 
}

void sendInitData( int nextGuy , int n , float* x , float* y , float* mass )
{
	int err = MPI_Send( &n , 1 , MPI_INT , nextGuy , 0 , MPI_COMM_WORLD ) ;
	if( err != MPI_SUCCESS )
		printf( "ERROR: failed init send1!\n" ) ;
	err = MPI_Send( x , n , MPI_FLOAT , nextGuy , 1 , MPI_COMM_WORLD ) ;
	if( err != MPI_SUCCESS )
		printf( "ERROR: failed init send2!\n" ) ;
	err = MPI_Send( y , n , MPI_FLOAT , nextGuy , 2 , MPI_COMM_WORLD ) ;
	if( err != MPI_SUCCESS )
		printf( "ERROR: failed init send3!\n" ) ;
	err = MPI_Send( mass , n , MPI_FLOAT , nextGuy , 3 , MPI_COMM_WORLD ) ;
	if( err != MPI_SUCCESS )
		printf( "ERROR: failed init send4!\n" ) ;
}

// UTILIZES MALLOC
void recvInitData( int from , int* n , float** x , float** y , float** mass )
{
	MPI_Status status ;
	MPI_Recv( n , 1 , MPI_INT , from , 0 , MPI_COMM_WORLD , &status ) ; 
	*x = (float*) malloc( (*n) * sizeof( float ) ) ; 
	*y = (float*) malloc( (*n) * sizeof( float ) ) ; 
	*mass = (float*) malloc( (*n) * sizeof( float ) ) ; 
	MPI_Recv( *x , *n , MPI_FLOAT , from , 1 , MPI_COMM_WORLD , &status ) ; 
	MPI_Recv( *y , *n , MPI_FLOAT , from , 2 , MPI_COMM_WORLD , &status ) ; 
	MPI_Recv( *mass , *n , MPI_FLOAT , from , 3 , MPI_COMM_WORLD , &status ) ; 
}

void sendUpdate( int nextGuy , float* x , float* y , int n , int i )
{
	int err = MPI_Send( x , n , MPI_FLOAT , nextGuy , 2*i + 4 , MPI_COMM_WORLD ) ;
	if( err != MPI_SUCCESS )
		printf( "ERROR: failed send1!\n" ) ;
	err = MPI_Send( y , n , MPI_FLOAT , nextGuy , 2*i + 5 , MPI_COMM_WORLD ) ;
	if( err != MPI_SUCCESS )
		printf( "ERROR: failed send2!\n" ) ;
}

void recvUpdate( int from , float* x , float* y , int n , int i )
{
	MPI_Status status ;
	int err = MPI_Recv( x , n , MPI_FLOAT , from , 2*i + 4 , MPI_COMM_WORLD , &status ) ;
	if( err != MPI_SUCCESS )
		printf( "ERROR: failed recv1!\n" ) ;
	err = MPI_Recv( y , n , MPI_FLOAT , from , 2*i + 5 , MPI_COMM_WORLD , &status ) ;
	if( err != MPI_SUCCESS )
		printf( "ERROR: failed recv2!\n" ) ;
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
	char temp1[1000] ;
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
//		printf( "%f %f %f\n" , x[j] , y[j] , mass[j] ) ;
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
//	printf( "converting '%s'\n" , temp ) ;
	return( atof( temp ) ) ;
}










