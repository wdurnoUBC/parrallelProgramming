
#include <stdlib.h>
#include <stdio.h>

#include "types.h"

struct nodeArg
{
	int pIdx ; 
	int part0 ; 
	int rows ; 
	int depth ; 
	int tree ; 
	int left ; // = 1 if this node is a left child, 0 otherwise 
}; 

void rand( int* seed )
{
        *seed = (*seed) * 1103515245 + 12345 ;
        if( *seed < 0 )
                *seed = -(*seed) ;
        *seed = (*seed) % 2147483648 ;
}

__device__ void d_rand( int* seed )
{
        *seed = (*seed) * 1103515245 + 12345 ;
        if( *seed < 0 )
                *seed = -(*seed) ;
        *seed = (*seed) % 2147483648 ;
}

__global__ void printData( float* data , int* participants , int* labels , int rows , int dims )
{
	printf( "Data:\n" ) ; 
	int i, j ; 
	for( i = 0 ; i < rows ; i++ ) 
	{
		printf( "%i: " , i ) ; 
		for( j = 0 ; j < dims ; j++ ) 
			printf( "%f " , data[ dims*i+j ] ) ; 
		printf( "\n" ) ; 
	}
	printf( "Participants:\n" ) ; 
	for( i = 0 ; i < rows ; i++ ) 
		printf( "%i " , participants[i] ) ; 
	printf( "\nlabels:\n" ) ; 
	for( i = 0 ; i < rows ; i++ )
		printf( "%i " , labels[i] ) ; 
	printf( "\n" ) ; 
} 

__global__ void kernel( node* nodes , nodeArg* args , nodeArg* children , int jobs , float* data , int* labels , int* participants, int* seeds, int labelMax, int* leftPart, int* rightPart, int* leftCounts, int* rightCounts , int breadth, int dim , int samp , int maxDepth )
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x ; 
	if( idx >= jobs ) 
		return ; 
	
	// assign default values 
	int i, j, k, leftAttr, rightAttr, leftErr , rightErr , pivotDim , bestPivotDim , leftBest , rightBest , bestErr , bestLeftClass , bestRightClass , bestLeftErr , bestRightErr ; 
	float del , bestPivotVal ; 
	bestPivotDim = -1 ; 
	bestPivotVal = 0.0 ; 
	d_rand( &(seeds[idx]) ) ; 
	leftAttr = seeds[idx] % (labelMax+1) ; 
	leftErr = 0 ;  
	leftBest = -1 ; 
	d_rand( &(seeds[idx]) ) ; 
	rightAttr = seeds[idx] % (labelMax+1) ; 
	rightErr = 0 ; 
	rightBest = -1 ; 
	bestErr = samp + 1 ; // max possible value plus one 
	bestLeftClass = leftAttr ; // random assignment 
	bestRightClass = rightAttr ; 
	bestLeftErr = 0 ; 
	bestRightErr = 0 ;  
	
	// find best pivot 
	for( i = 0 ; i < breadth ; i++ ) // cycle through test dimensions for potential pivot points 
	{
		d_rand( &(seeds[idx]) ) ; 
		pivotDim = seeds[idx] % dim ;
		for( j = 0 ; j < args[idx].rows ; j++ ) // cycle through data as for potential pivot points 
		{
			// set error counts to zero 
			for( k = 0 ; k < labelMax+1 ; k++ ) 
			{
				leftCounts[ idx * (labelMax+1) + k ] = 0 ; 
				rightCounts[ idx * (labelMax+1) + k ] = 0 ; 
			}
			
			// add up counts 
			for( k = 0 ; k < args[idx].rows ; k++ ) 
			{
				// del = data[ samp*dim*idx + dim*k + pivotDim ] - data[ samp*dim*idx + dim*j + pivotDim ] ; // test minus pivot  
				//( del < 0 ) ? ( leftCounts[ idx*(labelMax+1) + labels[samp*idx + k] ] += 1 ) : ( rightCounts[ idx*(labelMax+1) + labels[samp*idx + k] ] += 1 ) ; // increment counts 
				del = data[ dim*participants[ args[idx].part0 + k ] + pivotDim ] - data[ dim*participants[ args[idx].part0 + j ] + pivotDim ] ; 
				if( del < 0 ) 
					leftCounts[ idx*(labelMax+1) + labels[ participants[ args[idx].part0 + k ] ] ] += 1; 
				else
					rightCounts[ idx*(labelMax+1) + labels[ participants[ args[idx].part0 + k ] ] ] += 1 ; 
 			}
			
			/*
			// DEBUG LOOP
			printf( "pivotDim: %i , pivotRule: %f \n" , pivotDim , data[ samp*dim*idx + dim*j + pivotDim ] ) ;  
			for( k = 0 ; k < labelMax+1 ; k++ ) 
			{
				printf( "Counts for label %i: %i %i\n" , k , leftCounts[ idx*(labelMax+1) + k ] , rightCounts[ idx*(labelMax+1) + k ] ) ; 
			}
			return ; 
			*/
			
			leftBest = 0 ; 
			rightBest = 0 ; 
			leftErr = 0 ; 
			rightErr = 0 ;  
			for( k = 0 ; k < (labelMax+1) ; k++ ) 
			{
				if( leftCounts[idx*(labelMax+1) + k] > leftBest )
				{
					leftBest = leftCounts[idx*(labelMax+1) + k] ; 
					bestLeftClass = k ; 
				} 
				if( rightCounts[idx*(labelMax+1) + k] > rightBest ) 
				{
					rightBest = rightCounts[idx*(labelMax+1) + k] ; 
					bestRightClass = k ; 
				}
				leftErr += leftCounts[idx*(labelMax+1) + k] ; 
				rightErr += rightCounts[idx*(labelMax+1) + k] ; 
			}
			leftErr -= leftCounts[idx*(labelMax+1) + bestLeftClass] ; 
			rightErr -= rightCounts[idx*(labelMax+1) + bestRightClass] ; 
			
			// printf( "leftErr: %i, rightErr: %i, bestErr: %i\n" , leftErr , rightErr , bestErr ) ; 
			if( leftErr + rightErr < bestErr ) // find universal minimum 
			{
				leftAttr = bestLeftClass ; 
				rightAttr = bestRightClass ; 
				bestErr = leftErr + rightErr ; 
				bestPivotDim = pivotDim ; 
				bestPivotVal = data[samp*dim*idx + dim*j + pivotDim] ; 
				bestLeftErr = leftErr ; 
				bestRightErr = rightErr ; 
			}
		}
	}
	
	// write node 
	nodes[idx].dimension = bestPivotDim ; 
	nodes[idx].rule = bestPivotVal ; 
	nodes[idx].left = -1 ; // Default no child flag 
	nodes[idx].right = - 1 ; 
	nodes[idx].leftAttr = leftAttr ; 
	nodes[idx].rightAttr = rightAttr ; 
	
	/*
	// DEBUG 
	printf( "node[%i].dimension = %i \n" , idx , nodes[idx].dimension ) ; 
	printf( "node[%i].rule = %f \n" , idx , nodes[idx].rule ) ; 
	printf( "node[%i].left = %i \n" , idx , nodes[idx].left ) ; 
	printf( "node[%i].right = %i \n" , idx , nodes[idx].right ) ; 
	printf( "node[%i].leftAttr = %i \n" , idx , nodes[idx].leftAttr ) ; 
	printf( "node[%i].rightAttr = %i \n" , idx , nodes[idx].rightAttr ) ; 
	*/
	
	if( bestLeftErr == 0 && bestRightErr == 0 || args[idx].depth == maxDepth ) // Work is complete 
		return ; 
	
	// Rearrange participants for children 
	int leftPartN = 0 ; 
	int rightPartN = 0 ; 
	for( i = 0 ; i < args[idx].rows ; i++ ) // sort data into left & right parts 
	{
		if( data[ dim*participants[ args[idx].part0 + i ] + bestPivotDim ] < bestPivotVal ) // attribute left 
		{
			leftPart[ args[idx].part0 + leftPartN ] = participants[ args[idx].part0 + i ] ; 
			leftPartN += 1 ; 
		}
		else // attribute right 
		{
			rightPart[ args[idx].part0 + rightPartN ] = participants[ args[idx].part0 + i ] ; 
			rightPartN += 1 ; 
		}
	}
	for( i = 0 ; i < leftPartN ; i++ ) // reassign left part  
		participants[ args[idx].part0 + i ] = leftPart[ args[idx].part0 + i ] ; 
	for( i = 0 ; i < rightPartN ; i++ ) // reassign right part 
		participants[ args[idx].part0 + leftPartN + i ] = rightPart[ args[idx].part0 + i ] ; 
	
	/*
	// DEBUG 
	printf( "left child participants: " ) ; 
	for( i = 0 ; i < leftPartN ; i++ ) 
		printf( "%i " , participants[ args[idx].part0 + i ] ) ; 
	printf( "\nright child participants: " ) ; 
	for( i = 0 ; i < rightPartN ; i++ ) 
		printf( "%i " , participants[ args[idx].part0 + leftPartN + i ] ) ; 
	printf("\n") ; 
	*/
	
	// configure children requests 
	if( leftPartN > 0 && bestLeftErr > 0 ) // request left child request 
	{
		nodes[idx].left = 2*idx ; 
		children[2*idx].pIdx = idx ; 
		children[2*idx].part0 = args[idx].part0 ; 
		children[2*idx].rows = leftPartN ; 
		children[2*idx].depth = args[idx].depth + 1 ; 
		children[2*idx].tree = args[idx].tree ; 
		children[2*idx].left = 1 ; 
	}
	if( rightPartN > 0 && bestRightErr > 0 ) // request right child request 
	{
		nodes[idx].right = 2*idx+1 ; 
		children[2*idx+1].pIdx = idx ; 
		children[2*idx+1].part0 = args[idx].part0 + leftPartN ; 
		children[2*idx+1].rows = rightPartN ; 
		children[2*idx+1].depth = args[idx].depth + 1 ; 
		children[2*idx+1].tree = args[idx].tree ; 
		children[2*idx+1].left = 0 ; 
	}
	
	/*
	// DEBUG 
	if( leftPartN > 0 )
	{
		printf( "children[%i].pIdx = %i\n" , 2*idx , children[2*idx].pIdx ) ; 
		printf( "children[%i].part0 = %i\n" , 2*idx , children[2*idx].part0 ) ; 
		printf( "children[%i].rows = %i\n" , 2*idx , children[2*idx].rows ) ; 
		printf( "children[%i].depth = %i\n" , 2*idx , children[2*idx].depth ) ; 
	}
	if( rightPartN > 0 ) 
	{
		printf( "children[%i].pIdx = %i\n" , 2*idx+1 , children[2*idx+1].pIdx ) ; 
		printf( "children[%i].part0 = %i\n" , 2*idx+1 , children[2*idx+1].part0 ) ; 
		printf( "children[%i].rows = %i\n" , 2*idx+1 , children[2*idx+1].rows ) ; 
		printf( "children[%i].depth = %i\n" , 2*idx+1 , children[2*idx+1].depth ) ; 
	}
	*/
}

void forestTrain( node*** forest , int** nodeCount , int seed , int* labels , float** data , int rows , int dims , int breadth , int trees , int maxDepth , int samp )
{
	cudaDeviceReset() ; 
	
	*forest = (node**) malloc( trees * sizeof(node*) ) ;
        *nodeCount = (int*) malloc( trees * sizeof(int) ) ;
        int i ;
        for( i = 0 ; i < trees ; i++ )
        {
                (*forest)[i] = (node*) malloc( sizeof(node) ) ;
                (*nodeCount)[i] = 0 ;
        }

        // Allocate data on device 
        float* d_data ;
        int* d_participants ;
        int* d_labels ;
	int* d_leftPart ; 
	int* d_rightPart ; 
	
	cudaError_t status = cudaMalloc( &d_data , trees * samp * dims * sizeof(float) ) ;
        if( status == cudaSuccess )
                status = cudaMalloc( &d_participants , trees * samp * sizeof(int) ) ;
        if( status == cudaSuccess )
                status = cudaMalloc( &d_labels , trees * samp * sizeof(int) ) ; 
	if( status == cudaSuccess ) 
		status = cudaMalloc( &d_leftPart , trees * samp * sizeof(int) ) ; 
	if( status == cudaSuccess ) 
		status = cudaMalloc( &d_rightPart , trees * samp * sizeof(int) ) ; 
	
	int labelMax = 0 ;
        float* h_data = (float*) malloc( trees * samp * dims * sizeof(float) ) ; // CPU representation of compact GPU form of data
        int* h_participants = (int*) malloc( trees * samp * sizeof(int) ) ;
        int* h_labels = (int*) malloc( trees * samp * sizeof(int) ) ;

        int j , k , temp ;
        for( i = 0 ; i < trees && status == cudaSuccess ; i++ )
        {
                for( j = 0 ; j < samp && status == cudaSuccess ; j++ ) // Pick 'samp' random samples
                {
                        rand(&seed) ;
                        temp = seed % rows ; // the random row
                        for( k = 0 ; k < dims ; k++ )
                        {
                                h_data[ samp * dims * i + dims * j + k ] = data[temp][k] ; // copy data subset
                        }
                        h_labels[ samp * i + j ] = labels[ temp ] ;
                        h_participants[ samp * i + j ] = j ; // PARTICIPANTS IS DESIGNED TO SCALE! 
                        if( labelMax < labels[temp] )
                                labelMax = labels[temp] ;
                }
        }

        if( status == cudaSuccess ) 
                status = cudaMemcpy( d_data , h_data , trees * samp * dims * sizeof(float) , cudaMemcpyHostToDevice ) ; 
        if( status == cudaSuccess ) 
                status = cudaMemcpy( d_labels , h_labels , trees * samp * sizeof(int) , cudaMemcpyHostToDevice ) ; 
        if( status == cudaSuccess ) 
                status = cudaMemcpy( d_participants , h_participants , trees * samp * sizeof(int) , cudaMemcpyHostToDevice ) ; 
	
        free( h_data ) ;
        free( h_participants ) ;
        free( h_labels ) ;
	
	// Initialize temp variables 
	int jobs = trees ; 
	
	node* h_nodes ; 
	node* d_nodes ; 
	nodeArg* h_nodeReqs ; 
	nodeArg* d_nodeReqs ; 
	nodeArg* h_childReqs ; 
	nodeArg* d_childReqs ;
	int* h_seeds ; 
	int* d_seeds ; 
	int* d_leftCounts ; 
	int* d_rightCounts ; 
	
	h_nodes = (node*) malloc( jobs * sizeof(node) ) ; 
	h_nodeReqs = (nodeArg*) malloc( jobs * sizeof(nodeArg) ) ;  
	h_childReqs = (nodeArg*) malloc( 2 * jobs * sizeof(nodeArg) ) ; 
	h_seeds = (int*) malloc( jobs * sizeof(int) ) ; 
	for( i = 0 ; i < jobs ; i++ )
	{
		h_nodes[i].dimension = -1 ; 
		h_nodes[i].rule = 0.0 ; 
		h_nodes[i].left = -1 ; 
		h_nodes[i].right = -1 ; 
		h_nodes[i].leftAttr = -1 ; 
		h_nodes[i].rightAttr = -1 ; 
		
		h_nodeReqs[i].pIdx = -1 ; // no parent 
		h_nodeReqs[i].part0 = i * samp ; 
		h_nodeReqs[i].rows = samp ; 
		h_nodeReqs[i].depth = 1 ;
		h_nodeReqs[i].tree = i ; 
		h_nodeReqs[i].left = -1 ; 
		
		h_childReqs[2*i].pIdx = -1 ; // default flag for no request 
		h_childReqs[2*i+1].pIdx = -1 ; 
		h_childReqs[2*i].part0 = -1 ; 
		h_childReqs[2*i+1].part0 = -1 ; 
		h_childReqs[2*i].rows = -1 ; 
		h_childReqs[2*i+1].rows = -1 ; 
		h_childReqs[2*i].depth = -1 ; 
		h_childReqs[2*i+1].depth = -1 ; 
		h_childReqs[2*i].tree = -1 ; 
		h_childReqs[2*i+1].tree = -1 ; 
		h_childReqs[2*i].left = -1 ; 
		h_childReqs[2*i+1].left = -1 ; 
		
		temp = seed + i ; 
		rand( &temp ) ; 
		h_seeds[i] = temp ; 
	}
	
	if( status == cudaSuccess ) 
		status = cudaMalloc( &d_nodes , jobs * sizeof(node) ) ; 
	if( status == cudaSuccess )  
		status = cudaMemcpy( d_nodes , h_nodes , jobs * sizeof(node) , cudaMemcpyHostToDevice ) ; 
	if( status == cudaSuccess ) 
		status = cudaMalloc( &d_nodeReqs , jobs * sizeof(nodeArg) ) ; 
	if( status == cudaSuccess ) 
		status = cudaMemcpy( d_nodeReqs , h_nodeReqs , jobs * sizeof(nodeArg) , cudaMemcpyHostToDevice ) ; 
	if( status == cudaSuccess ) 
		status = cudaMalloc( &d_childReqs , 2 * jobs * sizeof(nodeArg) ) ; 
	if( status == cudaSuccess ) 
		status = cudaMemcpy( d_childReqs , h_childReqs , 2 * jobs * sizeof(nodeArg) , cudaMemcpyHostToDevice ) ; 
	if( status == cudaSuccess ) 
		status = cudaMalloc( &d_seeds , jobs * sizeof(int) ) ; 
	if( status == cudaSuccess ) 
		status = cudaMemcpy( d_seeds , h_seeds , jobs * sizeof(int) , cudaMemcpyHostToDevice ) ; 
	if( status == cudaSuccess ) 
		status = cudaMalloc( &d_leftCounts , jobs * (labelMax+1) * sizeof(int) ) ; 
	if( status == cudaSuccess ) 
		status = cudaMalloc( &d_rightCounts , jobs * (labelMax+1) * sizeof(int) ) ; 
	
	free( h_seeds ) ; 
	
	// printData<<< 1 , 1 >>>( d_data , d_participants , d_labels , trees * samp , dims ) ; 
	
	int blockSize = 32 ; // should be user-defined 
	int blocks = jobs/blockSize + 1 ; 
	node* tempNodes ; 
	nodeArg* tempNodeArgs ;  
	nodeArg* nextNodeReqs ; 
	int nextNodeReqsN ; 
	node* thisNode ; 
	
	// MAIN LOOP 
	while( jobs > 0 && status == cudaSuccess ) 
	{
		kernel<<< blocks , blockSize >>>( d_nodes , d_nodeReqs , d_childReqs , jobs , d_data , d_labels , d_participants , d_seeds , labelMax , d_leftPart , d_rightPart , d_leftCounts , d_rightCounts , breadth, dims , samp , maxDepth ) ;
		status = cudaDeviceSynchronize() ; 
		
		if( status == cudaSuccess ) 
			status = cudaMemcpy( h_nodes , d_nodes , jobs * sizeof(node) , cudaMemcpyDeviceToHost ) ; 
		if( status == cudaSuccess ) 
			status = cudaMemcpy( h_childReqs , d_childReqs , 2 * jobs * sizeof(nodeArg) , cudaMemcpyDeviceToHost ) ; 
		
		nextNodeReqs = (nodeArg*) malloc( sizeof(nodeArg) ) ; 
		nextNodeReqsN = 0 ; 
		for( i = 0 ; i < jobs ; i++ ) // load nodes into forest 
		{
			// store the new node 
			(*nodeCount)[ h_nodeReqs[i].tree ] += 1 ; // increment node total of this tree 
			tempNodes = (*forest)[ h_nodeReqs[i].tree ] ; // store old tree values 
			(*forest)[ h_nodeReqs[i].tree ] = (node*) malloc( (*nodeCount)[ h_nodeReqs[i].tree ] * sizeof(node) ) ; // malloc space for one more 
			memcpy( (*forest)[ h_nodeReqs[i].tree ] , tempNodes , ((*nodeCount)[ h_nodeReqs[i].tree ] - 1) * sizeof(node) ) ; // copy old values over 
			(*forest)[ h_nodeReqs[i].tree ][ (*nodeCount)[ h_nodeReqs[i].tree ] - 1 ] = h_nodes[i] ; // copy in the value of the new node 
			
			thisNode = &(  (*forest)[ h_nodeReqs[i].tree ][ (*nodeCount)[ h_nodeReqs[i].tree ] - 1 ]  ) ; 
			
			// if it has a parent, wire the parent to it 
			if( h_nodeReqs[i].pIdx >= 0 ) 
			{
				if( h_nodeReqs[i].left == 1 )
					(*forest)[ h_nodeReqs[i].tree ][ h_nodeReqs[i].pIdx ].left = (*nodeCount)[ h_nodeReqs[i].tree ] - 1 ; 
				else
					(*forest)[ h_nodeReqs[i].tree ][ h_nodeReqs[i].pIdx ].right = (*nodeCount)[ h_nodeReqs[i].tree ] - 1 ; 
			}
			
			// if it has child requests, fill them 
			if( thisNode->left >= 0 ) 
			{
				nextNodeReqsN += 1 ; // copy next node request into temp array  
				tempNodeArgs = nextNodeReqs ; 
				nextNodeReqs = (nodeArg*) malloc( nextNodeReqsN * sizeof(nodeArg) ) ; 
				memcpy( nextNodeReqs , tempNodeArgs , (nextNodeReqsN - 1) * sizeof(nodeArg) ) ; 
				free( tempNodeArgs ) ; 
				nextNodeReqs[ nextNodeReqsN - 1 ] = h_childReqs[ thisNode->left ] ; 
				nextNodeReqs[ nextNodeReqsN - 1 ].pIdx = (*nodeCount)[ h_nodeReqs[i].tree ] - 1 ; // assign pIdx to point to parent id 
			}
			if( thisNode->right >= 0 ) 
			{
				nextNodeReqsN += 1 ; 
				tempNodeArgs = nextNodeReqs ; 
				nextNodeReqs = (nodeArg*) malloc( nextNodeReqsN * sizeof(nodeArg) ) ; 
				memcpy( nextNodeReqs , tempNodeArgs , (nextNodeReqsN - 1) * sizeof(nodeArg) ) ; 
				free( tempNodeArgs ) ; 
				nextNodeReqs[ nextNodeReqsN - 1 ] = h_childReqs[ thisNode->right ] ; 
				nextNodeReqs[ nextNodeReqsN - 1 ].pIdx = (*nodeCount)[ h_nodeReqs[i].tree ] - 1 ; 
			}
		}
		
		
		// DEBUG 
		for( j = 0 ; j < trees ; j++ ) 
		{
			printf( "tree %i:\n" , j ) ; 
			for( k = 0 ; k < (*nodeCount)[j] ; k++ )
			{
				printf( "%i: dim: %i , rule: %f , left: %i , right: %i , leftAttr: %i , rightAttr: %i \n" , k , (*forest)[j][k].dimension , (*forest)[j][k].rule , (*forest)[j][k].left , (*forest)[j][k].right , (*forest)[j][k].leftAttr , (*forest)[j][k].rightAttr ) ; 
			}
		}
		
		
		// Assign the next batch of work 
		jobs = nextNodeReqsN ;  
		
		free( h_nodeReqs ) ; 
		free( h_childReqs ) ; 
		free( h_nodes ) ;
		cudaFree( d_nodeReqs ) ;  
		cudaFree( d_childReqs ) ; 
		cudaFree( d_nodes ) ; 
		
		if( jobs > 0 ) 
		{
			h_nodeReqs = nextNodeReqs ; 
			
			h_nodes = (node*) malloc( jobs * sizeof(node) ) ; 
			h_childReqs = (nodeArg*) malloc( 2 * jobs * sizeof(nodeArg) ) ; 
			
			// assign default values 
			for( i = 0 ; i < jobs ; i++ ) 
			{
				h_nodes[i].dimension = -1 ; 
				h_nodes[i].rule = 0.0 ; 
				h_nodes[i].left = -1 ; 
				h_nodes[i].right = -1 ; 
				h_nodes[i].leftAttr = -1 ; 
				h_nodes[i].rightAttr = -1 ; 
				
				h_childReqs[2*i].pIdx = -1 ; 
				h_childReqs[2*i+1].pIdx = -1 ; 
				h_childReqs[2*i].part0 = -1 ; 
				h_childReqs[2*i+1].part0 = -1 ; 
				h_childReqs[2*i].rows = -1 ; 
				h_childReqs[2*i+1].rows = -1 ; 
				h_childReqs[2*i].depth = -1 ; 
				h_childReqs[2*i+1].depth = -1 ; 
				h_childReqs[2*i].tree = -1 ; 
				h_childReqs[2*i+1].tree = -1 ; 
				h_childReqs[2*i].left = -1 ; 
				h_childReqs[2*i+1].left = -1 ;  
			}
			
			if( status == cudaSuccess ) 
				status = cudaMalloc( &d_nodeReqs , jobs * sizeof(nodeArg) ) ; 
			if( status == cudaSuccess ) 
				status = cudaMemcpy( d_nodeReqs , h_nodeReqs , jobs * sizeof(nodeArg) , cudaMemcpyHostToDevice ) ; 
			if( status == cudaSuccess ) 
				status = cudaMalloc( &d_nodes , jobs * sizeof(node) ) ; 
			if( status == cudaSuccess ) 
				status = cudaMemcpy( d_nodes , h_nodes , jobs * sizeof(node) , cudaMemcpyHostToDevice ) ; 
			if( status == cudaSuccess ) 
				status = cudaMalloc( &d_childReqs , 2 * jobs * sizeof(nodeArg) ) ; 
			if( status == cudaSuccess ) 
				status = cudaMemcpy( d_childReqs , h_childReqs , 2 * sizeof(nodeArg) , cudaMemcpyHostToDevice ) ; 
			
		}
		else
			free( nextNodeReqs ) ; 
		
	}
	
	if( status != cudaSuccess )
                printf( "ERROR: %s \n" , cudaGetErrorString(status) ) ; 
	
	cudaFree( d_data ) ;
        cudaFree( d_participants ) ;
        cudaFree( d_labels ) ;
	cudaFree( d_leftPart ) ; 
	cudaFree( d_rightPart ) ; 
	
	cudaDeviceReset() ; 
}
















