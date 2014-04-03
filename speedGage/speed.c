
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>

void master( int lambda ) ;
void slave() ;
int findPrimes( int n ) ;
int getRandomPoisson( double lambda ) ;

int main( int argc , char** argv )
{
	int lambda = 1234 ;
	
	if( argc > 1 )
		lambda = atoi( argv[1] ) ;
	
	int rank ;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	srand( time(NULL) + rank ) ;
	
	if( rank == 0 )
		master( lambda ) ;
	else
		slave() ;
	MPI_Finalize();
	return 0 ;
}

void master( int lambda )
{
	int n ;
	MPI_Comm_size (MPI_COMM_WORLD, &n);
	
	int work[ n-1 ] ;
	int i ;
	for( i = 1 ; i < n ; i++ ) // send work
	{
		work[i-1] = getRandomPoisson( lambda ) ;
		printf( "Sending %i to worker %i\n", work[i-1] , i ) ;
		int temp = work[i-1] ;
		int err = MPI_Send( &temp, 1 , MPI_INT , i , 0 , MPI_COMM_WORLD ) ;
		if( err != MPI_SUCCESS )
			printf( "MPI_Send err: %i\n" , err ) ;
	}
	
	int done = 0 ; // is all the work done yet?
	int doneWork[n-1] ;
	for( i = 1 ; i < n ; i++ )
		doneWork[i-1] = -1 ;
	
	MPI_Status status ;
	while( done == 0 ) // waiting for work completion
	{
		for( i = 1 ; i < n ; i++ )
		{
			MPI_Recv( &doneWork[i-1] , 1 , MPI_INT , i , 0 , MPI_COMM_WORLD , &status ) ;
		}
		
		done = 1 ;
		for( i = 1 ; i < n ; i++ )
		{
			if( doneWork[i-1] < 0 )
				done = 0 ;
		}
	}
	
	for( i = 1 ; i < n ; i++ )
	{
		printf( "Worker %i found all primes below %i in %i seconds\n" , i , work[i-1] , doneWork[i-1] ) ;
	}
}

void slave()
{
	int buff = -1 ;
	MPI_Status status ;
	while( buff < 0 )
	{
		MPI_Recv( &buff , 1 , MPI_INT , 0 , 0 , MPI_COMM_WORLD , &status ) ;
	}
	
	time_t tm = time(NULL) ;
	int primes = findPrimes( buff ) ;
	int out = (int) ( time(NULL) - tm ) ;
	printf( "%i primes below %i!\n" , primes , buff ) ;
	MPI_Send( &out , 1 , MPI_INT , 0 , 0 , MPI_COMM_WORLD ) ;
}

int findPrimes( int n )
{
	if( n < 2 )
		return( 0 ) ;
	if( n == 2 ) 
		return( 1 ) ;
	int i , j , isPrime ;
	int out = 1 ;
	for( i = 2 ; i < n ; i++ )
	{
		isPrime = 1 ;
		for( j = 2 ; (j < (int) sqrt(i)) && (isPrime != 0) ; j++ )
		{
			if( i % j == 0 )
				isPrime = 0 ;
		}
		if( isPrime != 0 )
			out = out + 1 ;
	}
	return( out ) ;
}

int getRandomPoisson( double lambda )
{
        int k = 1 ;
        double unif ;
        if( - lambda < log( DBL_MIN ) ) // utilize log scale to avoid roundoff errors, slower
        {
                double p = 0.0 ;
                while( p > - lambda )
                {
                        k = k + 1 ;
                        unif = ((double) rand() ) / ((double) RAND_MAX ) ;
                        p = p + log( unif ) ;
                }
        }
        else
        {
                double p = 1.0 ;
                while( p > exp(-lambda) )
                {
                        k = k + 1 ;
                        unif = ((double) rand() ) / ((double) RAND_MAX ) ;
                        p = p * unif ;
                }
        }
        return( k - 1 ) ;
}


