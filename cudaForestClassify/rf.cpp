
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "types.h"

struct node ;

struct node
{
	int dimension ; // data dimension to make the decision at
	double rule ; // if greater than or equal to rule, go right. Otherwise, left
	node* left ;
	node* right ;
	int attributeLeft ; // if left node is null, access this value
	int attributeRight ;
};

int treeAttribute( node* root , double** data, int datumRow ) ;
void rand( int* seed ) ;
void classErr( double** data , int* labels , int labelMax , int rows , int partDim , int partRow , int* participants, int* greaterClassification , int* lesserClassification , int* lesserErr , int* greaterErr ) ;
void nodeTrain( int* seed , int* labels , int labelMax , double** data , int rows , int dims , node* root , int breadth , int* participants , int depth , int maxDepth , int samp ) ; 
void treeTrain( node** root , int* seed , int* labels , double** data , int rows , int dims , int breadth , int maxDepth , int samp ) ; 
void forestTrain( node*** forest , int seed , int* labels , double** data , int rows , int dims , int breadth , int trees , int maxDepth , int samp ) ; 
int forestClassify( node** forest , int trees , int labelMax , double** data , int datumRow ) ;
int treeToStringHelper( node* tree , char** out ) ;
char* treeToString( node* tree ) ;
void deleteTree( node* tree ) ;
node* stringToTreeHelper( char** str , int* pos ) ;
node* stringToTree( char* serialTree ) ;
char** forestToString( node** forest , int trees ) ; 
int stringToForest( char** str , node*** forest , int trees ) ; 
void forestEval( node** forest , int trees , int* labels , double** data , int rows , int dims , int*** attMat , double** err ) ; 
void printEval( int labelMax , int** attMat , double* err ) ; 
void serializeForest( node** forest , int trees , cuNode** cuNodes , int** starts , int* length ) ; 

struct randomForest // I don't actually use this...
{
	int n ; // forest size
	node* trees ; // array of root nodes, representing individual trees.
};

int treeAttribute( node* root , double** data, int datumRow )
{
	if( data[datumRow][ root->dimension ] < root->rule ) // go left
	{
		if( root->left == NULL )
			return( root->attributeLeft ) ;
		else
			return( treeAttribute( root->left , data , datumRow ) ) ;
	}
	else // go right
	{
		if( root->right == NULL )
			return( root->attributeRight ) ;
		else
			return( treeAttribute( root->right , data , datumRow ) ) ;
	}
}


void rand( int* seed )
{
	*seed = (*seed) * 1103515245 + 12345 ;
	if( *seed < 0 )
		*seed = -(*seed) ;
	*seed = (*seed) % 2147483648 ;
}

// partDim : dimension of division
// part : partition between classification regions
// greaterClassification : output, general vote value for values greater than partition
// lesserClassification : output, general vote value for values less than partition
// lesserErr : total errors due to general vote for values less than partition
// greaterErr : total errors due to general vote for values greater than partition
void classErr( double** data , int* labels , int labelMax , int rows , int partDim , int partRow , int* participants, int* greaterClassification , int* lesserClassification , int* lesserErr , int* greaterErr )
{
	int leftCounts[ labelMax+1 ] ;
	int rightCounts[ labelMax+1 ] ;
	int i ;
	for( i = 0 ; i < labelMax+1 ; i++ ) 
	{
		leftCounts[i] = 0 ;
		rightCounts[i] = 0 ;
	}
	for( i = 0 ; i < rows ; i++ )
	{
		if( data[ participants[i] ][ partDim ] < data[ participants[partRow] ][partDim] )
			leftCounts[ labels[ participants[i] ] ] += 1 ;
		else
			rightCounts[ labels[ participants[i] ] ] += 1 ;
	}
	int leftBest = 0 ; 
	*lesserClassification = 0 ;
	*lesserErr = 0 ;
	int rightBest = 0 ; 
	*greaterClassification = 0 ;
	*greaterErr = 0 ;
	for( i = 0 ; i < labelMax+1 ; i++ ) 
	{
		if( leftCounts[i] > leftBest )
		{
			leftBest = leftCounts[i] ;
			*lesserClassification = i ;
		}
		if( rightCounts[i] > rightBest )
		{
			rightBest = rightCounts[i] ;
			*greaterClassification = i ;
		}
		*lesserErr += leftCounts[i] ;
		*greaterErr += rightCounts[i] ;
	}
	if( *lesserErr == 0 )
		*lesserClassification = labels[ participants[partRow] ] ; 
	else
		*lesserErr -= leftCounts[ *lesserClassification ] ; 
	*greaterErr -= rightCounts[ *greaterClassification ] ;
}

// NOTE: 'rows' will be re-used to denote the length of 'participants'
// TODO Needless recursion is techinically possible--implement information gain to correct
void nodeTrain( int* seed , int* labels , int labelMax , double** data , int rows , int dims , node* root , int breadth , int* participants , int depth , int maxDepth , int samp ) // helper
{
	double mass[labelMax+1] ;
	int temp , i , j , k ;
	int bestDim = -1 ;
	double bestErr , bestSplit , bestLeftErr , bestRightErr , bestLeftClass, bestRightClass ;
	int leftClass, rightClass, leftErr , rightErr ;
	for( i = 0 ; i < breadth ; i++ ) // try dimensions
	{
		rand( seed ) ;
		temp = (*seed) % dims ; // select test dimension
		if( rows > samp ) // If too many, take a random subset
			rows = samp + 1 ; 
		for( j = 0 ; j < rows ; j++ ) 
		{
			// get next split
			if( rows > samp )
			{
				rand( seed ) ; 
				k = (*seed) % rows ; 
			}
			else // If few enough samples, go thru all
				k = j ; 
			
			// calculate errors
			classErr( data , labels , labelMax , rows , temp , k , participants, &rightClass , &leftClass , &leftErr , &rightErr ) ; 
			if( bestDim < 0 )
			{
				bestDim = temp ;
				bestSplit = data[ participants[k] ][temp] ; 
				bestErr = leftErr + rightErr ; 
				bestLeftErr = leftErr ; 
				bestRightErr = rightErr ; 
				bestLeftClass = leftClass ; 
				bestRightClass = rightClass ; 
			}
			if( bestErr > leftErr + rightErr ) 
			{
				bestDim = temp ;
				bestSplit = data[ participants[k] ][temp] ;
				bestErr = leftErr + rightErr ;
				bestLeftErr = leftErr ; 
				bestRightErr = rightErr ; 
				bestLeftClass = leftClass ; 
				bestRightClass = rightClass ; 
			}
		}
	}
	root->dimension = bestDim ;
	root->rule = bestSplit ;
	root->attributeLeft = bestLeftClass ;
	root->attributeRight = bestRightClass ;
	leftErr = bestLeftErr ; 
	rightErr = bestRightErr ; 
	int* leftPart = (int*) malloc( 1 * sizeof(int) ) ;
	int* rightPart = (int*) malloc( 1 * sizeof(int) ) ;
	int leftPartN = 0 ;
	int rightPartN = 0 ;
	if( leftErr != 0 || rightErr != 0 )
	{
		for( i = 0 ; i < rows ; i++ )
		{
			if( data[ participants[i] ][bestDim] < bestSplit ) 
			{
				leftPartN += 1 ;
				leftPart = (int*) realloc( leftPart , leftPartN * sizeof(int) ) ;
				leftPart[ leftPartN-1 ] = participants[i] ;
			}
			else
			{
				rightPartN += 1 ;
				rightPart = (int*) realloc( rightPart , rightPartN * sizeof(int) ) ;
				rightPart[ rightPartN-1 ] = participants[i] ; 
			}
		}
	}
	if( leftErr == 0 )
		root->left = NULL ;
	else
	{
		if( depth < maxDepth )
		{
			root->left = (node*) malloc( sizeof(node) ) ; 
			nodeTrain( seed , labels , labelMax , data , leftPartN , dims , root->left , breadth , leftPart , depth+1 , maxDepth , samp ) ; 
		}
		else
			root->left = NULL ; 
	}
	free( leftPart ) ;
	if( rightErr == 0 )
		root->right = NULL ;
	else
	{
		if( depth < maxDepth ) 
		{
			root->right = (node*) malloc( sizeof(node) ) ;
			nodeTrain( seed , labels , labelMax , data , rightPartN , dims , root->right , breadth , rightPart , depth+1 , maxDepth , samp ) ;
		}
		else
			root->right = NULL ; 
	}
	free( rightPart ) ;
}

// trains a random tree
// seed : a random number seed, good for parallelizing tree building
// labels : values to classify for, non-negative, count from zero. I WON'T WASTE TIME CHECKING FOR THIS!
// data : matrix of data, rows represent data, columns represent dimensions (dims)
// breadth : number of random dimensions to try (with replacement) for minimizing entropy
void treeTrain( node** root , int* seed , int* labels , double** data , int rows , int dims , int breadth , int maxDepth , int samp ) // interface
{
	*root = (node*) malloc( sizeof(node) ) ; ;
	int participants[ rows ] ;
	int labelMax = 0 ;
	int i ;
	for( i = 0 ; i < rows ; i++ )
	{
		participants[i] = i ; 
		if( labels[i] > labelMax )
			labelMax = labels[i] ;
	}
	nodeTrain( seed , labels , labelMax , data , rows , dims , *root , breadth ,  participants , 1 , maxDepth , samp ) ;
}

// returns a list of nodes of length 'trees'
// UTILIZES MALLOC!
void forestTrain( node*** forest , int seed , int* labels , double** data , int rows , int dims , int breadth , int trees , int maxDepth , int samp )
{
	(*forest) = (node**) malloc( trees * sizeof(node*) ) ;
	int seeds[trees] ; 
	int i ; 
	for( i = 0 ; i < trees ; i++ ) // TODO PARALLELIZE THIS 
	{
		printf( "Training tree %i of %i\n" , i , trees ) ; 
		seeds[i] = seed + i ; 
		treeTrain( &(*forest)[i] , &seeds[i] , labels , data , rows , dims , breadth , maxDepth , samp ) ;
	}
}

int forestClassify( node** forest , int trees , int labelMax , double** data , int datumRow )
{
	int counts[ labelMax+1 ] ;
	int i ;
	for( i = 0 ; i < labelMax+1 ; i++ )
		counts[i] = 0 ;
	int classification ;
	for( i = 0 ; i < trees ; i++ )
	{
		classification = treeAttribute( forest[i] , data , datumRow ) ; 
		counts[ classification ] += 1 ; 
	}
	int best = 0 ;
	int max = counts[0] ;
	for( i = 1 ; i < labelMax+1 ; i++ )
	{
		if( counts[i] > max )
		{
			best = i ;
			max = counts[i] ;
		}
	}
	return( best ) ;
}


int treeToStringHelper( node* tree , char** out )
{
	char temp[1000] ;
	int len1 = sprintf( temp , "(%i,%f,%i,%i," , tree->dimension , tree->rule , tree->attributeLeft , tree->attributeRight ) ; 
	int len2 = strlen( *out ) ;
	*out = (char*) realloc( *out , (len1 + len2 + 1)*sizeof(char) ) ; 
	strcat( *out , temp ) ;
	if( tree->left == NULL )
	{
		len1 = strlen( *out ) ;
		*out = (char*) realloc( *out , (len1 + 5 + 1)*sizeof(char) ) ; 
		strcat( *out , "NULL," ) ;
	}
	else
	{
		treeToStringHelper( tree->left , out ) ;
		len1 = strlen( *out ) ;
		*out = (char*) realloc( *out , (len1 + 2)*sizeof(char) ) ; 
		strcat( *out , "," ) ; 
	}
	if( tree->right == NULL )
	{
		len1 = strlen( *out ) ; 
		*out = (char*) realloc( *out , (len1 + 5 + 1)*sizeof(char) ) ; 
		strcat( *out , "NULL)" ) ; 
	}
	else
	{
		treeToStringHelper( tree->right , out ) ; 
		len1 = strlen( *out ) ; 
		*out = (char*) realloc( *out , (len1 + 2)*sizeof(char) ) ; 
		strcat( *out , ")" ) ; 
	}
}

char* treeToString( node* tree )
{
	char* out = (char*) malloc( sizeof(char) ) ;
	out[0] = '\0' ; 
	treeToStringHelper( tree , &out ) ; 
	return( out ) ; 
}

void deleteTree( node* tree )
{
	if( tree == NULL )
		return ; 
	if( tree->left != NULL )
		deleteTree( tree->left ) ;
	if( tree->right != NULL )
		deleteTree( tree->right ) ; 
	free( tree ) ; 
}

// returns NULL if malformed
node* stringToTreeHelper( char** str , int* pos )
{
	int len = strlen( *str ) ; 
	int temp1 , temp2 ;
	char tempString[100] ; 
	if( ! (*pos < len) )
		return( NULL ) ; 
	if( (*str)[*pos] == '(' ) 
	{
		temp1 = *pos ; 
		int i ;
		node tempNode ; 
		for( i = *pos+1 ; i < len && temp1 == *pos ; i++ ) // get first datum 
		{
			if( (*str)[i] == ',' ) 
				temp1 = i ; 
		}
		temp2 = temp1 ; 
		for( i = temp1+1 ; i < len && temp2 == temp1 ; i++ ) // get second datum 
		{
			if( (*str)[i] == ',' ) 
				temp2 = i ;
		}
		if( temp1 == *pos || temp1 == temp2 ) // malformed data
			return( NULL ) ; 
		
		for( i = *pos + 1 ; i < temp1 ; i++ ) // read strings into memory
			tempString[ i - (*pos)-1 ] = (*str)[i] ; 
		tempString[ temp1 - (*pos) - 1 ] = '\0' ; 
		tempNode.dimension = atoi( tempString ) ; 
		for( i = temp1 + 1 ; i < temp2 ; i++ )
			tempString[ i - temp1 - 1 ] = (*str)[i] ; 
		tempString[ temp2 - temp1 - 1 ] = '\0' ; 
		tempNode.rule = atof( tempString ) ; 
		
		temp1 = temp2 ;
		for( i = temp1 + 1 ; i < len && temp2 == temp1 ; i++ ) // get third datum 
		{
			if( (*str)[i] == ',' ) 
				temp2 = i ; 
		}
		if( temp1 == temp2 ) // malformed data 
			return( NULL ) ; 
		
		for( i = temp1 + 1 ; i < temp2 ; i++ ) // read string into memory 
			tempString[ i - temp1 - 1 ] = (*str)[i] ; 
		tempString[ temp2 - temp1 - 1 ] = '\0' ; 
		tempNode.attributeLeft = atoi( tempString ) ; 
		
		temp1 = temp2 ; 
		for( i = temp1 + 1 ; i < len && temp2 == temp1 ; i++ ) // get fourth datum 
		{
			if( (*str)[i] == ',' ) 
				temp2 = i ;
		}
		if( temp1 == temp2 ) // malformed data 
			return( NULL ) ; 
		
		for( i = temp1 + 1 ; i < temp2 ; i++ ) // read string into memory 
			tempString[ i - temp1 - 1 ] = (*str)[i] ; 
		tempString[ temp2 - temp1 - 1 ] = '\0' ; 
		tempNode.attributeRight = atoi( tempString ) ; 
		
		*pos = temp2 + 1 ; 
		
		node* left = stringToTreeHelper( str , pos ) ; 
		if( (*str)[(*pos)+1] != ',' )
		{
			deleteTree( left ) ; 
			return( NULL ) ; 
		}
		*pos += 2 ;
		
		node* right = stringToTreeHelper( str , pos ) ; 
		if( (*str)[(*pos)+1] != ')' ) 
		{
			deleteTree( left ) ; 
			deleteTree( right ) ; 
			return( NULL ) ; 
		}
		
		*pos = (*pos) + 1 ; 
		
		node* out = (node*) malloc( sizeof(node) ) ; 
		out->dimension = tempNode.dimension ; 
		out->rule = tempNode.rule ; 
		out->left = left ; 
		out->right = right ; 
		out->attributeLeft = tempNode.attributeLeft ; 
		out->attributeRight = tempNode.attributeRight ; 
	}
	else
	{
		if( (*str)[*pos] == 'N' && (*str)[*pos+1] == 'U' && (*str)[*pos+2] == 'L' && (*str)[*pos+3] == 'L' ) 
			*pos += 3 ;
		// otherwise is malformed data 
		return( NULL ) ;
	}
}

node* stringToTree( char* serialTree )
{
	int pos = 0 ; 
	node* out = stringToTreeHelper( &serialTree , &pos ) ; 
	return( out ) ; 
}


// UTILIZES MALLOC
// IF storing on disk, write each row as a line
char** forestToString( node** forest , int trees )
{
	char** out = (char**) malloc( trees * sizeof(char*) ) ;
	int i ; 
	for( i = 0 ; i < trees ; i++ ) 
	{
		out[i] = treeToString( forest[i] ) ; 
	}
	return( out ) ; 
}

// returns the number of trees in the forest
// forest is an output parameter, UTILIZES MALLOC!
int stringToForest( char** str , node*** forest , int trees )
{
	*forest = (node**) malloc( trees * sizeof(node*) ) ; 
	int i ; 
	for( i = 0 ; i < trees ; i++ ) 
	{
		(*forest)[i] = stringToTree( str[i] ) ; 
	}
}

// attMat : matrix of attribution counts. Rows are attributed as columns
// err : proportion of errors within the attribution class
// UTILIZES MALLOC
void forestEval( node** forest , int trees , int* labels , double** data , int rows , int dims , int*** attMat , double** err )
{
	int i , j ; 
	int labelMax = 0 ; 
	for( i = 0 ; i < rows ; i++ ) 
	{
		if( labels[i] > labelMax ) 
			labelMax = labels[i] ; 	
	}
	
	*attMat = (int**) malloc( (labelMax+1) * sizeof(int*) ) ; 
	for( i = 0 ; i < labelMax+1 ; i++ )
		(*attMat)[i] = (int*) malloc( (labelMax+1)*sizeof(int) ) ; 
	*err = (double*) malloc( (labelMax+1) * sizeof(double) ) ; 
	for( i = 0 ; i < labelMax+1 ; i++ )
	{
		(*err)[i] = 0.0 ; 
		for( j = 0 ; j < labelMax+1 ; j++ ) 
			(*attMat)[i][j] = 0 ; 
	}
	
	int att ; 
	for( i = 0 ; i < rows ; i++ ) 
	{
		att = forestClassify( forest , trees , labelMax , data , i ) ; 
		(*attMat)[ labels[i] ][att] += 1 ; 
		(*err)[ labels[i] ] += 1.0 ; 
	}
	
	double temp ; 
	for( i = 0 ; i < labelMax+1 ; i++ ) 
	{
		temp = (*err)[i] ;
		if( temp > 0.0 ) 
		{ 
			(*err)[i] = (*err)[i] - ((double) (*attMat)[i][i]) ; 
			(*err)[i] = (*err)[i] / temp ; 
		}
	}
}

void printEval( int labelMax , int** attMat , double* err )
{
	int i , j ; 
	for( i = 0 ; i < labelMax+1 ; i++ ) 
	{
		for( j = 0 ; j < labelMax+1 ; j++ ) 
			printf( "%i " , attMat[i][j] ) ;
		printf( " err: %f" , err[i] ) ;  
		printf("\n") ; 
	}
}

int serializeForestHelperAddNode( node* thisNode , cuNode** cuNodes , int* length )
{
	*cuNodes = (cuNode*) realloc( *cuNodes , (*length + 1) * sizeof(cuNode) ) ; 
	(*cuNodes)[*length].dim = thisNode->dimension ; 
	(*cuNodes)[*length].rule = thisNode->rule ; 
	(*cuNodes)[*length].left = -2 ; 
	(*cuNodes)[*length].right = -2 ; 
	(*cuNodes)[*length].leftAttr = thisNode->attributeLeft ; 
	(*cuNodes)[*length].rightAttr = thisNode->attributeRight ; 
	*length += 1 ;
	return( *length-1 ) ; 
}

void serializeForestHelper( node* tree , int thisCuNode , cuNode** out , int* length ) 
{
	if( tree->left == NULL ) 
		(*out)[ thisCuNode ].left = -1 ; 
	else
	{
		int temp1 = serializeForestHelperAddNode( tree->left , out , length ) ; 
		(*out)[thisCuNode].left = temp1 ; 
		serializeForestHelper( tree->left , (*out)[thisCuNode].left , out , length ) ; 
	}
	
	if( tree->right == NULL ) 
		(*out)[ thisCuNode ].right = -1 ; 
	else
	{
		// (*out)[ thisCuNode ].right = serializeForestHelperAddNode( tree->right , out , length ) ; 
		int temp2 = serializeForestHelperAddNode( tree->right , out , length ) ; 
		(*out)[thisCuNode].right = temp2 ; // I have no idea why this temp variable is necessary, but it is 
		serializeForestHelper( tree->right , (*out)[thisCuNode].right , out , length ) ; 
	}
}

void serializeForest( node** forest , int trees , cuNode** cuNodes , int** starts , int* length )
{
	*cuNodes = (cuNode*) malloc( sizeof(cuNode) ) ; 
	*length = 0 ; 
	*starts = (int*) malloc( trees * sizeof(int) ) ; 
	
	int i , thisCuNode ; 
	for( i = 0 ; i < trees ; i++ ) 
	{
		(*starts)[i] = *length ; 
		thisCuNode = serializeForestHelperAddNode( forest[i] , cuNodes , length ) ; 
		serializeForestHelper( forest[i] , thisCuNode , cuNodes , length ) ; 
	}
}
























