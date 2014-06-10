#include "math.h"
#include "bvh.h"
#include "types.h"
__device__ int Theta(uint32 i, uint32 j, uint32* keys, uint32 size)
{
	if (j<0 || j>size)
			return -1;
		uint32 a1 = keys[i];
		uint32 a2 = keys[j];
		if ( a1==a2)
		{ 
			a1 = i;
			a2 = j;
		}
		uint32 a = a1^a2;
		
		return __nlz(a); 
}
__global__ void buildKernel(Bvh_Node* nodes, Bvh_Node* leaves, uint32* keys);
