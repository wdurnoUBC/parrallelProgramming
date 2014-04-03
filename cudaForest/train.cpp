
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
using namespace std ; 

#include "types.h"

//extern struct node ; 
extern void forestTrain( node*** forest , int** nodeCount , int seed , int* labels , float** data , int rows , int dims , int breadth , int trees , int maxDepth , int samp ) ; 
// extern void forestEval( node** forest , int trees , int* labels , float** data , int rows , int dims , int*** attMat , float** err ) ;
// extern void printEval( int labelMax , int** attMat , float* err ) ; 
// extern char** forestToString( node** forest , int trees ) ; 

bool readData( string fileName , float*** matrix , int* rows , int* cols )
{
        ifstream file( fileName.c_str() ) ;
        if( ! file.is_open() )
                return false ;
        *matrix = (float**) malloc( sizeof(float*) ) ;
        *rows = 0 ;
        string s ;
        float val ;
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
		*matrix = (float**) realloc( *matrix , (*rows) * sizeof(float*) ) ;
		(*matrix)[0] = (float*) malloc( sizeof(float) ) ; 
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
				(*matrix)[0] = (float*) realloc( (*matrix)[0] , (*cols) * sizeof(float) ) ; 
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
		*matrix = (float**) realloc( *matrix , (*rows) * sizeof(float*) ) ; 
		(*matrix)[(*rows)-1] = (float*) malloc( (*cols) * sizeof(float) ) ; 
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

bool readLabels( string fileName , int** labels , int* n )
{
	ifstream file( fileName.c_str() ) ;
	if( ! file.is_open() )
		return false ;
	*labels = (int*) malloc( sizeof(int) ) ; 
	*n = 0 ; 
	string s ; 
	while( file.good() ) 
	{
		getline( file , s ) ; 
		*n += 1 ; 
		*labels = (int*) realloc( *labels , (*n) * sizeof(int) ) ; 
		(*labels)[ (*n) - 1 ] = atoi( s.c_str() ) ; 
	}
	file.close() ; 
	return true ; 
}

int main( int argc , char** argv )
{
	// init
	if( argc < 5 )
	{
		cout << "Please provide a label file (arg1), a data matrix (arg2), a forest output file (arg3), a random sample size (arg4), and, optionally, an integer seed (arg5)" << endl ; 
		return -1 ; 
	}
	
	int seed = 1 ;
	if( argc > 5 )
		seed = atoi( argv[5] ) ; 
	
	// Load in data
	cout << "loading data..." << endl ; 
	int* labels ;  
	float** data ; 
	int rows ; 
	int cols ; 
	
	readLabels( string(argv[1]) , &labels , &rows ) ; 
	
	readData( string(argv[2]) , &data , &rows , &cols ) ; 
	cout << " data has " << rows << " rows and " << cols << " dims" << endl ;
	
	// Train forest
	cout << "Training forest..." << endl ; 
	node** forest ;
	int breadth = 45 ; 
	int trees = 10 ; //71 ; // TODO replace this values with real inputs
	int maxDepth = 60 ; 
	int samp = atoi(argv[4]) ; 
	int* nodeCount ; 
	forestTrain( &forest , &nodeCount , seed , labels , data , rows , cols , breadth , trees , maxDepth , samp ) ; 
	
	/*
	char** str = forestToString( forest , trees ) ;
        for( int i = 0 ; i < trees ; i++ )
        {
                printf( str[i] ) ;
                printf( "\n" ) ;
        }
	*/
	
	/*
	// Evaluate forest
	int** attMat ; 
	float* err ; 
	forestEval( forest , trees , labels , data , rows , cols , &attMat , &err ) ;
	int maxLab = -1 ;
	for( int i = 0 ; i < rows ; i++ )
	{
		if( labels[i] > maxLab )
			maxLab = labels[i] ; 
	}
	cout << "Evaluating forest on " << maxLab << " labels..." << endl ; 
	printEval( maxLab , attMat , err ) ; 
	
	
	// Print the forest
	cout << "Printing forest to disk" << endl ; 
	char** str = forestToString( forest , trees ) ; 
	ofstream outFile( argv[3] ) ; 
	for( int i = 0 ; i < trees ; i++ )
	{
		outFile << string(str[i]) << endl ; 
	}
	outFile.close() ; 
	*/
	
	
	cout << "done" << endl ; 
	
	return( 0 ) ; 
}


