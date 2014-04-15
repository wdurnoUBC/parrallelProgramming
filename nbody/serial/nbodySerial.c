
#include <time.h>
#include <sys/time.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

double getVal( char* , int , int ) ;
void loadData( char* fileName , int n , double* x , double* y , double* mass , int* actual ) ;
void updateLocalVars( int rank , int n , double* xVel , double* yVel ,  double* x , double* y , double* mass , double G , double delt , double* xOut , double* yOut ) ; 

int main( int argc , char** argv )
{
	int rank ;
	
	if( argc < 3 && rank == 0 )
	{
		printf( "Please provide the name (arg1) of a space-delimited file storing a real-valued matrix with a positive number of rows, 3 columns, and the last column positive.\n" ) ;
		printf( "Please also pass a positive integer (arg2) indicating the number of simulation cycles\n" ) ;
		printf( "Please provide (arg3) the number of bodies in the N-body simulation.\n" ) ;
		printf( "Optionally, provide (arg4) a time-step size, by default delt = 1000 s\n" ) ; 
		return( 1 ) ;
	}
	
	double G = 6.67384 * pow( 10.0 , -11.0 ) ;
	double delt = 1000 ; 
	if( argc > 4 ) 
		delt = atof( argv[4] ) ; 
	
	int n = atoi( argv[3] ) ; 
	int m ; 
	int iterSize = atoi( argv[2] ) ; 
	double x[n] ; 
	double y[n] ; 
	double xVel[n] ; 
	double yVel[n] ; 
	double mass[n] ; 
	double xTemp[n] ; 
	double yTemp[n] ; 
	loadData( argv[1] , n , x , y , mass , &m ) ; 
	
	int i , j ;
	for( i = 0 ; i < m ; i++ ) 
	{
		xVel[i] = 0.0 ; 
		yVel[i] = 0.0 ; 
	}
	
	// start timer 
	struct timeval start , end ; 
	gettimeofday( &start , NULL ) ; 
	
	// start work 
	for( i = 0 ; i < iterSize ; i += 2 ) 
	{
		for( j = 0 ; j < m ; j++ ) 
			updateLocalVars( j , m , xVel , yVel , x , y , mass , G , delt , xTemp , yTemp ) ; 
		for( j = 0 ; j < m ; j++ ) 
			updateLocalVars( j , m , xVel , yVel , xTemp , yTemp , mass , G , delt , x , y ) ; 
	} // end work
	
	// end timer 
	gettimeofday( &end , NULL ) ; 
	fprintf( stderr , "CPU time: %ld microseconds\n", ((end.tv_sec * 1000000 + end.tv_usec)
                  - (start.tv_sec * 1000000 + start.tv_usec))) ; 
	
	// Work complete 
	for( i = 0 ; i < n ; i++ ) 
		printf( "%f\t%f\n" , x[i] , y[i] ) ; 
	
	return( 0 ) ;
}

void updateLocalVars( int rank , int n , double* xVel , double* yVel ,  double* x , double* y , double* mass , double G , double delt , double* xOut , double* yOut )
{
	int i ;
	double r ;
	double xForce = 0.0 ; 
	double yForce = 0.0 ; 
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

void loadData( char* fileName , int n , double* x , double* y , double* mass , int* actual )
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

double getVal( char* str , int start , int subLen )
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










