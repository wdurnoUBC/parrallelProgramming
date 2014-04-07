
struct cuNode
{
	int dim ; 
	float rule ; 
	int left ; 
	int right ; 
	int leftAttr ; 
	int rightAttr ; 
};

struct cudaForest
{
        cuNode* cuNodes ;
        int* starts ;
        int* gpuTempSpace ;
        int maxLabel ;
        int trees ;
};

