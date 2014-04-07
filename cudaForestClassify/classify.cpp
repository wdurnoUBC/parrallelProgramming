
#include <time.h>
#include <sys/time.h>

#include "rf.cpp"

#include <string.h>
#include <iostream>
#include <fstream>
#include <string.h>
using namespace std ; 

extern void loadDataToGPU( cuNode* cuNodes , int cuNodesN , int* starts , int trees , float* datum , int dims ) ; 

bool readData( string fileName , double*** matrix , int* rows , int* cols )
{
        ifstream file( fileName.c_str() ) ;
        if( ! file.is_open() )
                return false ;
        *matrix = (double**) malloc( sizeof(double*) ) ;
        *rows = 0 ;
        string s ;
        double val ;
        int space1 , space2 ;
	*cols = 0 ; 
	// count columns
	if( file.good() )
	{
		getline( file , s ) ; 
		int n = s.length() ;
		if( n == 0 ) 
			return false ; 
		*rows += 1 ; 
		*matrix = (double**) realloc( *matrix , (*rows) * sizeof(double*) ) ;
		(*matrix)[0] = (double*) malloc( sizeof(double) ) ; 
		space1 = 0 ; 
		for( int i = 0 ; i < n && space1 == 0 ; i++ )
		{
			if( s[i] == ' ' ) 
				space1 = i ; 
		}
		*cols = 1 ;
		(*matrix)[0][0] = atof( s.substr(0,space1).c_str() ) ; 
		
		for( int i = space1+1 ; i < n ; i++ )
		{
			if( s[i] == ' ' ) 
			{
				space2 = i ; 
				*cols += 1 ; 
				(*matrix)[0] = (double*) realloc( (*matrix)[0] , (*cols) * sizeof(double) ) ; 
				(*matrix)[0][(*cols)-1] = atof( s.substr(space1+1,space2).c_str() ) ; 
				space1 = space2 ; 
				i = space1 + 1 ; 
			}
		}
	}
	
	// Read in following rows
        while( file.good() ) 
        {
		getline( file , s ) ; 
		int n = s.length() ; 
		int col = 0 ;
		if( n == 0 )
			break ; 
		*rows += 1 ; 
		*matrix = (double**) realloc( *matrix , (*rows) * sizeof(double*) ) ; 
		(*matrix)[(*rows)-1] = (double*) malloc( (*cols) * sizeof(double) ) ; 
		space1 = 0 ; 
		for( int i = 0 ; i < n && space1 == 0 ; i++ ) 
		{
			if( s[i] == ' ' ) 
				space1 = i ; 
		}
		(*matrix)[(*rows)-1][col] = atof( s.substr(0,space1).c_str() ) ; 
		
		for( int i = space1 + 1 ; i < n && col+1 < *cols ; i++ ) 
		{
			if( s[i] == ' ' ) 
			{
				col += 1 ; 
				space2 = i ; 
				(*matrix)[(*rows)-1][col] = atof( s.substr(space1+1,space2).c_str() ) ; 
				space1 = space2 ; 
				i = space1 + 1 ; 
			}
		}
        }
	file.close() ; 
        return true ;
}


int main( int argc , char** argv )
{
	// init
	if( argc < 4 )
	{
		cout << "Please provide a label file (arg1), a data matrix (arg2), and forest unput file (arg3)." << endl ; 
		return -1 ; 
	}
	
	int seed = 1 ;
	if( argc > 5 )
		seed = atoi( argv[5] ) ; 
	
	// Load in data
	cout << "loading data..." << endl ; 
	int* labels ;  
	double** data ; 
	int rows ; 
	int cols ; 
	
	readData( string(argv[1]) , &data , &rows , &cols ) ; 
	if( rows < 1 || cols < 1 )
	{
		cout << "Malformed label file" << endl ; 
		return -1 ; 
	}
	
	labels = (int*) malloc( rows * sizeof(int) ) ; 
	for( int i = 0 ; i < rows ; i++ )
	{
		labels[i] = (int) data[i][0] ; 
		free( data[i] ) ; 
	}
	free(data) ; 
	
	readData( string(argv[2]) , &data , &rows , &cols ) ; 
	cout << " data has " << rows << " rows and " << cols << " dims" << endl ;
	
	/*
	// Train forest
	cout << "Training forest..." << endl ; 
	node** forest ;
	int breadth = 50 ; 
	int trees = 10 ; 
	int maxDepth = 60 ; 
	int samp = atoi(argv[4]) ; 
	
	if( argc > 6 )
		trees = atoi( argv[6] ) ; 
	
	s
	// DEBUG truct timeval start, end ;
	gettimeofday(&start, NULL) ;
	
	forestTrain( &forest , seed , labels , data , rows , cols , breadth , trees , maxDepth , samp ) ; 
	
	gettimeofday(&end, NULL) ;
	
	printf("%ld microseconds\n", ((end.tv_sec * 1000000 + end.tv_usec)
                  - (start.tv_sec * 1000000 + start.tv_usec))) ; 
	*/
	
	node** forest = (node**) malloc( sizeof(node*) ) ; 
	int trees = 0 ; 
	
	ifstream forestFile( argv[3] ) ; 
	string tempString ; 
	char* tempCharAr ; 
	while( forestFile.good() ) 
	{
		getline( forestFile , tempString ) ; 
		trees += 1 ; 
		forest = (node**) realloc( forest , trees * sizeof(node*) ) ; 
		tempCharAr = (char*) malloc( (tempString.length() + 1) * sizeof(char) ) ; 
		strcpy( tempCharAr , tempString.c_str() ) ; 
		// printf( "%s\n" ,  tempCharAr ) ; // Data is read correctly 
		forest[trees-1] = stringToTree( tempCharAr ) ; 
		free(tempCharAr) ; 
	}
	trees -= 1 ; 
	
	/*
	// DEBUG
	char** str = forestToString( forest , trees ) ; 
        for( int i = 0 ; i < trees ; i++ )
        {
                printf( "%s\n" , str[i] ) ;
                printf( "\n" ) ;
        }
	*/
	
	cuNode* cuNodes ; 
	int* starts ; 
	int length ; 
	serializeForest( forest , trees , &cuNodes , &starts , &length ) ; 
	
	float datum[cols] ; 
	for( int i = 0 ; i < cols ; i++ ) 
		datum[i] = data[0][i] ; 
	
	loadDataToGPU( cuNodes , length , starts , trees , datum , cols ) ; 
	
	/*
	// DEBUG 
	for( int i = 0 ; i < trees ; i++ ) 
	{
		int lim = length ; 
		if( i < trees - 1 ) 
			lim = starts[i+1] ; 
		printf( "Tree %i:\n" , i ) ; 
		for( int j = starts[i] ; j < lim ; j++ ) 
		{
			printf( "%i: dim: %i, rule: %f, left: %i, right: %i, leftAttr: %i, rightAttr: %i \n" , j , cuNodes[j].dim , cuNodes[j].rule , cuNodes[j].left , cuNodes[j].right , cuNodes[j].leftAttr , cuNodes[j].rightAttr ) ; 
		}
	}
	*/
	
	struct timeval start, end ;
	gettimeofday(&start, NULL);
	int fCl = forestClassify( forest , trees , 2 , data , 0 ) ; 
	gettimeofday(&end, NULL);
	cout << "CPU Forest classification: " << fCl << endl ; 
	
	printf("CPU attribution time: %ld\n", ((end.tv_sec * 1000000 + end.tv_usec)
		  - (start.tv_sec * 1000000 + start.tv_usec)));
	
	// Evaluate forest
	cout << "Evaluating forest..." << endl ; 
	int** attMat ; 
	double* err ; 
	forestEval( forest , trees , labels , data , rows , cols , &attMat , &err ) ; 
	printEval( 2 , attMat , err ) ; 
	
	cout << "done" << endl ; 
	
	return( 0 ) ; 
}












