#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "cuda_klbvh.h"
#include "MortonCode.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "objLoader.h"
#include <algorithm>
#include <cstdlib>
#include "klbvh.h"
#include "gputimer.cuh"
#include "timer.h"
#include <cub/cub.cuh>
using namespace nih;
int globalCounter = 0;
texture<float> BboxTex;
texture<uint32> bvhTex;

uint32 GridSize(uint32 count)
{
	int numSMs;
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
	size_t max_blocks = 65535;
	size_t n_blocks   = nih::min( max_blocks, (count + (128*numSMs)-1) / (128*numSMs) );
	return numSMs*n_blocks;
}

struct bvhTexHelper
{
	static const uint32 nodeSize = 11;
	static const uint32 LChildOf = 6;
	static const uint32 RChildOf = 7;
	static const uint32 pidOf = 8;
	static const uint32 leafStartOf = 9;
	static const uint32 leafEndOf = 10;
	float p[6];
	NIH_DEVICE float* getBbox(uint32 id)
	{
		uint32 offset = id*nodeSize;
		uint32 t[6];
		
		for(int i=0; i<6; i++)
		{
			t[i] = tex1Dfetch(bvhTex,offset+i);
			p[i] = bitsToFloat(t[i]);
		}

	
		return p;
	}

	NIH_DEVICE uint32 getLChild(uint32 id)
	{
		return tex1Dfetch(bvhTex,id*nodeSize+LChildOf);
	}

	NIH_DEVICE uint32 getRChild(uint32 id)
	{
		return tex1Dfetch(bvhTex,id*nodeSize+RChildOf);
	}
	NIH_DEVICE uint32 getPid(uint32 id)
	{
		return tex1Dfetch(bvhTex,id*nodeSize+pidOf);
	}
	NIH_DEVICE uint32 getleafStart(uint32 id)
	{
		return tex1Dfetch(bvhTex,id*nodeSize+leafStartOf);
	}
	NIH_DEVICE uint32 getleafEnd(uint32 id)
	{
		return tex1Dfetch(bvhTex,id*nodeSize+leafEndOf);
	}
	NIH_DEVICE bool isLeaf(uint32 id)
	{
		return getleafStart(id) == getleafEnd(id);
	}
};

//FORCE_INLINE NIH_DEVICE void traverseIterative(Bbox3f& qbox, uint32 qId,uint32 leafIdx,
//	cuda::DBVH* bvh,
//	uint32* list)
//{
//	Bvh_Node* stack[64];
//	uint32 top = 0;
//	stack[top++] = bvh->getRoot();
//	while(top>0)
//	{
//		Bvh_Node* node = stack[--top];
//		if (node->leafEnd <= leafIdx)
//			continue;
//		Bbox3f box = bvh->getNodeBox(node);
//		
//		bool ret = AABBOverlap(qbox, box);
//		if (ret )
//		{
//			//相交
//			if (node->l_isleaf)
//			{
//				Bvh_Node* leaf = (bvh->getLLeafChild(node));
//				Bbox3f box = bvh->getLeafBox(leaf);
//				if (AABBOverlap(qbox,box))
//				{
//					list->add(Pair(qId,leaf->pid));
//				}				
//			}
//			else
//				stack[top++] = bvh->getLChild(node);
//			if(node->r_isleaf)
//			{
//				Bvh_Node* leaf = (bvh->getRLeafChild(node));
//				Bbox3f box = bvh->getLeafBox(leaf);
//				if (AABBOverlap(qbox,box))
//				{
//					list->add(Pair(qId,leaf->pid));
//				}				
//			}
//			else
//				stack[top++] =  bvh->getRChild(node);
//		}		
//	}
//}
FORCE_INLINE NIH_DEVICE void FrustumCulling(pyrfrustum_t& frustum, uint32 frustumId,
	cuda::DBVH* bvh,uint32 priSize,
	uint32* list)
{
	Bvh_Node* stack[64];
	uint32 top = 0;
	stack[top++] = bvh->getRoot();
	uint32 offset = priSize*frustumId;
	while(top>0)
	{
		Bvh_Node* node = stack[--top];
		Bbox3f box = bvh->getNodeBox(node);
		
		int ret = Intersect(frustum, box);
		if (ret == 2)
		{
			//相交
			if (node->l_isleaf)
			{
				Bvh_Node* leaf = (bvh->getLLeafChild(node));
				Bbox3f box = bvh->getLeafBox(leaf);
				if (Intersect(frustum,box))
				{
					list[offset+leaf->pid] = 1;
				}				
			}
			else
				stack[top++] = bvh->getLChild(node);
			if(node->r_isleaf)
			{
				Bvh_Node* leaf = (bvh->getRLeafChild(node));
				Bbox3f box = bvh->getLeafBox(leaf);
				if (Intersect(frustum,box))
				{
					list[offset+leaf->pid] = 1;
				}				
			}
			else
				stack[top++] =  bvh->getRChild(node);
		}
		else if (ret == 1)
		{
			//in
			for(int k= node->leafStart; k<=node->leafEnd;k++)
				list[offset+bvh->leaves[k].pid] = 1;
		}
	}
}
FORCE_INLINE NIH_DEVICE void FrustumCulling(TriFrustum& frustum, uint32 frustumId,
	cuda::DBVH* bvh,uint32 priSize,
	uint32* list)	
{
	Bvh_Node* stack[64];
	uint32 top = 0;
	stack[top++] = bvh->getRoot();
	uint32 offset = priSize*frustumId;
	while(top>0)
	{
		Bvh_Node* node = stack[--top];
		Bbox3f box = bvh->getNodeBox(node);
		
		int ret = Intersect(frustum, box);
		if (ret == 2)
		{
			//相交
			if (node->l_isleaf)
			{
				Bvh_Node* leaf = (bvh->getLLeafChild(node));
				Bbox3f box = bvh->getLeafBox(leaf);
				if (Intersect(frustum,box))
				{
					list[offset+leaf->pid] = 1;
				}				
			}
			else
				stack[top++] = bvh->getLChild(node);
			if(node->r_isleaf)
			{
				Bvh_Node* leaf = (bvh->getRLeafChild(node));
				Bbox3f box = bvh->getLeafBox(leaf);
				if (Intersect(frustum,box))
				{
					list[offset+leaf->pid] = 1;
				}				
			}
			else
				stack[top++] =  bvh->getRChild(node);
		}
		else if (ret == 1)
		{
			//in
			for(int k= node->leafStart; k<=node->leafEnd;k++)
				list[offset+bvh->leaves[k].pid] = 1;
		}
	}
}
FORCE_INLINE NIH_DEVICE void FrustumCulling(pyrfrustum_t& frustum, uint32 frustumId,
	Bintree* bvh,uint32 priSize,
	uint32* out)
{
	uint32 offset = priSize*frustumId;
	const uint32 stack_size  = 64;
	uint32 stack[stack_size];
	uint32 top = 0;
	stack[top++] = 0;
	while(top>0)
	{
		uint32 idx = stack[--top];		
		int ret = Intersect(frustum,bvh->boxPtr[idx]);
		if (ret == 2)
		{
			//相交
			
			if(bvh->isLeafPtr[bvh->RChildPtr[idx]])
			{
				if (Intersect(frustum,bvh->boxPtr[bvh->RChildPtr[idx]]))
				{
					out[offset+bvh->pidPtr[bvh->RChildPtr[idx]]] = 1;					
				}				
			}
			else
				stack[top++] = bvh->RChildPtr[idx];

			if (bvh->isLeafPtr[bvh->LChildPtr[idx]])
			{
				
				if (Intersect(frustum,bvh->boxPtr[bvh->LChildPtr[idx]]))
				{
					out[offset+bvh->pidPtr[bvh->LChildPtr[idx]]] = 1;					
				}				
			}
			else
				stack[top++] = bvh->LChildPtr[idx];
		}
		else if (ret == 1)
		{
			//in
			for(int k= bvh->leafStartPtr[idx]; k<=bvh->leafEndPtr[idx];k++)
				out[offset+offset+bvh->pidPtr[k]] = 1;
			
		}
	}
}
FORCE_INLINE NIH_DEVICE void FrustumCulling(pyrfrustum_t& frustum, uint32 frustumId,
	Bintree_Node* bvh,uint32 priSize,
	uint32* out)
{
	uint32 offset = priSize*frustumId;
	const uint32 stack_size  = 64;
	Bintree_Node* stack[stack_size];
	Bintree_Node** stackPtr = stack;
	*stackPtr++ = NULL;
	Bintree_Node* node = &bvh[0];
	do
    {
        // Check each child node for overlap.
		Bintree_Node* childL = &bvh[node->lChild];
		Bintree_Node* childR = &bvh[node->RChild];
        int overlapL = ( Intersect(frustum, 
			node->lBox) );
        int overlapR = ( Intersect(frustum, 
			node->rBox) );

        // Query overlaps a leaf node => report collision.
		if (overlapL && bvh[node->lChild].isLeaf())
			out[offset + bvh[node->lChild].leafStart] = 1;

		if (overlapR && bvh[node->RChild].isLeaf())
            out[offset + bvh[node->RChild].leafStart] = 1;
		
        // Query overlaps an internal node => traverse.
        bool traverseL = false;
		if (overlapL == 1)
		{
			for(int k= childL->leafStart; k<=childL->leafEnd; k++)
				out[offset + k] = 1;
		}
		else if( overlapL == 2 && !bvh[node->lChild].isLeaf())
		{
			traverseL = true;
		}
        bool traverseR = false;
		if (overlapR == 1)
		{
			for(int k= childR->leafStart; k<=childR->leafEnd; k++)
				out[offset + k] = 1;
		}
		else if( overlapR == 2 && !bvh[node->RChild].isLeaf())
		{
			traverseR = true;
		}

        if (!traverseL && !traverseR)
            node = *--stackPtr; // pop
        else
        {
            node = (traverseL) ? childL : childR;
            if (traverseL && traverseR)
                *stackPtr++ = childR; // push
        }
    }
    while (node != NULL);
}
FORCE_INLINE NIH_DEVICE void FrustumCulling(pyrfrustum_t& frustum, uint32 frustumId,
	Bintree_node* bvh,uint32 priSize,
	uint32* out)
{
	bvhTexHelper helper;
	uint32 offset = priSize*frustumId;
	const uint32 stack_size  = 64;
	uint32 stack[stack_size];
	uint32 top = 0;
	stack[top++] = 0;
	while(top>0)
	{
		uint32 idx = stack[--top];
		//Bintree_node * node = &bvh[idx];
		uint32 RChild = helper.getRChild(idx);
		uint32 LChild = helper.getLChild(idx);
		
		int ret = Intersect(frustum,helper.getBbox(idx));
		if (ret == 2)
		{
			//相交
			
			if(helper.isLeaf(RChild))
			{
				if (Intersect(frustum,helper.getBbox(RChild)))
				{
					out[offset+helper.getPid(RChild)] = 1;					
				}				
			}
			else
				stack[top++] = RChild;

			if (helper.isLeaf(LChild))
			{
				
				if (Intersect(frustum,helper.getBbox(LChild)))
				{
					out[offset+helper.getPid(LChild)] = 1;					
				}				
			}
			else
				stack[top++] = LChild;
		}
		else if (ret == 1)
		{
			//in
			for(int k= helper.getleafStart(idx); k<=helper.getleafEnd(idx);k++)
				out[offset+k] = 1;
		}
	}
}


//FORCE_INLINE NIH_DEVICE void FrustumCulling(pyrfrustum_t& frustum, uint32 frustumId,
//	uint32 priSize,
//	uint32* out)
//{
//	uint32 offset = priSize*frustumId;
//	const uint32 stack_size  = 64;
//	const uint32 nodeOffset = 64;
//	uint32 stack[stack_size];
//	uint32 top = 0;
//	stack[top++] = 0;
//	while(top>0)
//	{
//		uint32 idx = stack[--top];
//		Bintree_node * node = tex1Dfetch(bvhTex,idx*64);
//		int ret = Intersect(frustum,node->box);
//		if (ret == 2)
//		{
//			//相交
//			
//			if(bvh[node->RChild].isLeaf())
//			{
//				if (Intersect(frustum,bvh[node->RChild].box))
//				{
//					out[offset+bvh[node->RChild].pid] = 1;					
//				}				
//			}
//			else
//				stack[top++] = node->RChild;
//
//			if (bvh[node->lChild].isLeaf())
//			{
//				
//				if (Intersect(frustum,bvh[node->lChild].box))
//				{
//					out[offset+bvh[node->lChild].pid] = 1;					
//				}				
//			}
//			else
//				stack[top++] = node->lChild;
//		}
//		else if (ret == 1)
//		{
//			//in
//			for(int k= node->leafStart; k<=node->leafEnd;k++)
//				out[offset+k] = 1;
//		}
//	}
//}
//FORCE_INLINE NIH_DEVICE void FrustumCullingT(pyrfrustum_t& frustum, uint32 frustumId,
//	Bintree_node* bvh,uint32 priSize,
//	uint32* out)
//{
//	Bintree_node* stack[64];
//    Bintree_node** stackPtr = stack;
//    *stackPtr++ = NULL; // push
//	uint32 offset = priSize*frustumId;
//    // Traverse nodes starting from the root.
//    Bintree_node* node = &bvh[0];
//    do
//    {
//        // Check each child node for overlap.
//		Bintree_node* childL = &bvh[node->lChild];
//        Bintree_node* childR = &bvh[node->RChild];
//        int overlapL = ( Intersect(frustum, 
//			childL->box) );
//        int overlapR = (Intersect(frustum, 
//			childR->box) );
//
//        // Query overlaps a leaf node => report collision.
//		if (overlapL>0 && childL->isLeaf())
//			out[offset+ childL->pid] = 1;
//
//		if (overlapR>0 && childR->isLeaf())
//            out[offset+ childR->pid] = 1;
//
//		/*if (overlapL == 1 && !childL->isLeaf)
//		{
//			for(int k=childL->leafStart; k<=childL->leafEnd;k++)
//				out[offset+k] = 1;
//		}
//
//		if (overlapR == 1 && !childR->isLeaf )
//		{
//			for(int k=childR->leafStart; k<=childR->leafEnd;k++)
//				out[offset+k] = 1;
//		}*/
//        // Query overlaps an internal node => traverse.
//        bool traverseL = (overlapL == 2 && !childL->isLeaf());
//        bool traverseR = (overlapR ==2 && !childR->isLeaf());
//
//        if (!traverseL && !traverseR)
//            node = *--stackPtr; // pop
//        else
//        {
//            node = (traverseL) ? childL : childR;
//            if (traverseL && traverseR)
//                *stackPtr++ = childR; // push
//        }
//    }
//    while (node != NULL);
//}
__global__ void BruteforceFrustumCullingKernel(pyrfrustum_t* frustum, Bbox3f* boxes, uint32 priSize,uint32* list)
{
	uint32 step = blockDim.x * gridDim.x;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
		i < priSize; 
		i += step) 
	{
		int offset = i*6;
		float p[6];
		p[0] = tex1Dfetch(BboxTex,offset);
		p[1] = tex1Dfetch(BboxTex,offset+1);
		p[2] = tex1Dfetch(BboxTex,offset+2);
		p[3] = tex1Dfetch(BboxTex,offset+3);
		p[4] = tex1Dfetch(BboxTex,offset+4);
		p[5] = tex1Dfetch(BboxTex,offset+5);
		
	
		if (Intersect(*frustum,p) >0 )
			list[i] = 1;
	}
}
__global__ void FrustumCullingKernel(pyrfrustum_t* frustumP, int frustum_num, Bintree_Node* bvh, uint32 priSize,uint32* list)
{
	uint32 step = blockDim.x * gridDim.x;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
		i < frustum_num; 
		i += step) 
	{
		pyrfrustum_t frustum = frustumP[i];
		FrustumCulling(frustum,i,bvh,priSize,list);
		//FrustumCullingT(frustum,i,bvh,priSize,list);
	}
}
__global__ void FrustumCullingKernel(pyrfrustum_t* frustumP, int frustum_num, Bintree_node* bvh, uint32 priSize,uint32* list)
{
	uint32 step = blockDim.x * gridDim.x;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
		i < frustum_num; 
		i += step) 
	{
		pyrfrustum_t frustum = frustumP[i];
		FrustumCulling(frustum,i,bvh,priSize,list);
		//FrustumCullingT(frustum,i,bvh,priSize,list);
	}
}
__global__ void FrustumCullingKernel(pyrfrustum_t* frustumP, int frustum_num, Bintree* bvh, uint32 priSize,uint32* list)
{
	uint32 step = blockDim.x * gridDim.x;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
		i < frustum_num; 
		i += step) 
	{
		pyrfrustum_t frustum = frustumP[i];
		FrustumCulling(frustum,i,bvh,priSize,list);
		//FrustumCullingT(frustum,i,bvh,priSize,list);
	}
}
__global__ void FrustumCullingKernel(TriFrustum* frustumP, int frustum_num, cuda::DBVH* bvh, uint32 priSize,uint32* list)
{
	uint32 step = blockDim.x * gridDim.x;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
		i < frustum_num; 
		i += step) 
	{
		TriFrustum frustum = frustumP[i];
		FrustumCulling(frustum,i,bvh,priSize,list);
	}
}
__global__ void FrustumCullingKernel(pyrfrustum_t* frustumP, int frustum_num, cuda::DBVH* bvh, uint32 priSize,uint32* list)
{
	uint32 step = blockDim.x * gridDim.x;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
		i < frustum_num; 
		i += step) 
	{
		pyrfrustum_t frustum = frustumP[i];
		FrustumCulling(frustum,i,bvh,priSize,list);
	}
}

bool NIH_HOST_DEVICE GenerateVirFrustum(uint32 id, const Vector3f& eye,const Vector3f& p1,const Vector3f& p2, const Vector3f& p3, float farD, TriFrustum& frustum)
{
	     //求5个平面方程
		//视锥平面法线指向视锥外
		plane_t pTri(p1,p2,p3);	
		
		float d  = pTri.distance(eye);
		//视点不能位于三角形平面法线那一侧
		if (d<= 0)
			return false;

		//求虚视点
		Vector3f fNormal(pTri.a,pTri.b,pTri.c);
		float dir = dot(eye-p1,fNormal);
		Vector3f vEye = eye-fNormal*2.f*dir;
		
		frustum.id = id;
		frustum.planes[0] = plane_t(vEye,p2,p1);
		frustum.planes[1] = plane_t(vEye,p3,p2);
		frustum.planes[2] = plane_t(vEye,p1,p3);
		frustum.planes[3] =  plane_t(p1,p3,p2);
		frustum.planes[4] = pTri;
		Vector3f c = (p1+p2+p3)*1.f/3.f;
		float cosT = d/euclidean_distance(vEye,c);
		frustum.planes[4].d -= farD*cosT;		
		Vector3f p4,p5,p6;
		frustum.planes[4].intersect(vEye,p1,p4);
		frustum.planes[4].intersect(vEye,p2,p5);
		frustum.planes[4].intersect(vEye,p3,p6);
		frustum.center = (p1+p2+p3+p4+p5+p6)*1.f/6.f;
		frustum.min = nih::min(p1,p2);
		frustum.min = nih::min(frustum.min,p3);
		frustum.min = nih::min(frustum.min,p4);
		frustum.min = nih::min(frustum.min,p5);
		frustum.min = nih::min(frustum.min,p6);

		frustum.max = nih::max(p1,p2);
		frustum.max = nih::max(frustum.max,p3);
		frustum.max = nih::max(frustum.max,p4);
		frustum.max = nih::max(frustum.max,p5);
		frustum.max = nih::max(frustum.max,p6);
		return true;
}

//虚视锥生成kernel
__global__ void GenerateVirFrustumKernel(Vector3f* eye,Vector3f* p123, TriFrustum* frustums,float farD, int count)
{
	uint32 step = blockDim.x * gridDim.x;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
		i < count; 
		i += step) 
	{
		GenerateVirFrustum(i,*eye,p123[i*3],p123[i*3+1],p123[i*3+2],farD,frustums[i]);
		
	}
}

//包围盒裁剪，计算在包围盒内的三角形列表
//@Box 包围盒
//@TrianglePoints 三角形（每个三角形3个点）
//@size 三角形数量
//@list 在包围盒内的三角形id
__global__ void BboxCullingKernel(Bbox3f* Box, Vector3f* TrianglePoints, uint32 size, char* list)
{
	uint32 step = blockDim.x * gridDim.x;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
		i < size; 
		i += step) 
	{
		uint32 offset = 3*i;
		uint32 in = 0;
		for(int j=0; j<3; j++)
		{
			Vector3f p = TrianglePoints[offset+j];
			if (contains(*Box,p))
				in++;
		}
		
		if (in > 0)
		{
			list[offset] = 1;
			list[offset+1] = 1;
			list[offset+2] = 1;

		}
		
	}
}

__global__ void PrepareBvhPointKernel(Vector3f* TrianglePoints, Vector3f* centers,Bbox3f* boxes, uint32 size)
{
	uint32 step = blockDim.x * gridDim.x;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
		i < size; 
		i += step) 
	{
		uint32 offset = i*3;
		centers[i] = (TrianglePoints[offset] +TrianglePoints[offset+1] +TrianglePoints[offset+2])/3.f;
		boxes[i] = Bbox3f(TrianglePoints[offset]);
		boxes[i].insert(TrianglePoints[offset+1]);
		boxes[i].insert(TrianglePoints[offset+2]);
	
	}
}
__global__ void PrepareBvhPointKernel(TriFrustum* frustums, Vector3f* centers,Bbox3f* boxes, uint32 size)
{
	uint32 step = blockDim.x * gridDim.x;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
		i < size; 
		i += step) 
	{
		
		centers[i] = frustums[i].center;
		boxes[i] = Bbox3f(frustums[i].min,frustums[i].max);
		
	
	}
}

__global__ void FrustumCullingKernel(TriFrustum* frustums,cuda::DBVH* d_bvh,uint32 frustumNum, uint32 leavesNum,uint32* list)
{
	uint32 step = blockDim.x * gridDim.x;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
		i < leavesNum; 
		i += step) 
	{
		if (d_bvh->leaves[i].pid >= leavesNum)
		{
			TriFrustum frustum = frustums[d_bvh->leaves[i].pid - leavesNum];
			FrustumCulling(frustum,d_bvh->leaves[i].pid - leavesNum,d_bvh,leavesNum,list);
		}
	}
}
//__global__ void CollisonDetectionKernel(cuda::DBVH* bvh, cuda::SimpleList<Pair>* list, int size)
//{
//	int idx = threadIdx.x + blockDim.x * blockIdx.x;
//    if (idx < size)
//    {
//        Bbox3f qbox = bvh->leafBoxes[idx];
//		uint32 qId = bvh->leaves[idx].pid;
//        traverseIterative(qbox,qId, idx, bvh,list); 
//                      
//    }
//}



void CpuKlbvhTest(const thrust::host_vector<Vector3f>& h_points,
	const thrust::host_vector<Bbox3f>& h_boxes, const Bbox3f& gbox,
	pyrfrustum_t& frustum, BVH& bvh);
void GpuKlbvhTest(const thrust::host_vector<Vector3f>& h_points,
	const thrust::host_vector<Bbox3f>& h_boxes, 
	const Bbox3f& gBox,
	pyrfrustum_t& frustum, 
	thrust::host_vector<Bvh_Node>& h_nodes,
	thrust::host_vector<Bvh_Node>& h_leaves,
	thrust::host_vector<Bbox3f>& h_nodeBoxes,
	thrust::host_vector<Bbox3f>& h_leafBoxes
	)
{
	
	size_t size = h_points.size();
	thrust::device_vector<Vector3f> d_points = h_points;
	thrust::device_vector<Bbox3f> d_boxes = h_boxes;

	//{
	//random data
	/*size = 12;
	thrust::host_vector<Vector3f> points(size);
	thrust::host_vector<Bbox3f> boxes(size);

	for(int i =0; i< points.size(); i++)
	{
	points[i] = Vector3f(1.0*(rand()%100)/100,1.0*(rand()%100)/100,1.0*(rand()%100)/100);
	printVector3f(points[i]);
	boxes[i].insert(points[i]);
	gBox.insert(boxes[i]);
	}	
	d_points = points;
	d_boxes = boxes;*/
	//}
	//load from obj file
	/*size = loadObj(objFileName,d_points,d_boxes,gBox);*/
	thrust::device_vector<Bvh_Node> nodes(size-1);
	thrust::device_vector<Bvh_Node> leaves(size);
	cub::CachingDeviceAllocator allocator(true);
	cuda::KBvh_Builder builder(nodes,leaves,allocator);

	/*cub::DoubleBuffer<uint32> d_codes;
	size_t n_points = points.size();
	allocator.DeviceAllocate((void**)&d_codes.d_buffers[0], sizeof(uint32) * n_points);
	thrust::device_ptr<uint32> d_ptr(d_codes.d_buffers[0]);
	thrust::transform(
	d_points.begin(),
	d_points.begin() + n_points,
	d_ptr,
	morton_functor<uint32>( gBox ) );
	uint32 * h_codes = (uint32*)malloc(sizeof(uint32)*n_points);
	cudaMemcpy(h_codes,d_codes.d_buffers[0],sizeof(uint32)*n_points,cudaMemcpyDeviceToHost);
	for(int i=0; i<n_points; i++)
	std::cout<<h_codes[i]<<std::endl;*/
	cuda::DBVH h_bvh;
	/*cudaMalloc((void**)&d_bvh,sizeof(cuda::DBVH));*/
	GpuTimer timer;
	timer.Start();
	builder.build(gBox,d_points.begin(),d_points.end(),d_boxes.begin(),d_boxes.end(),&h_bvh);
	timer.Stop();
	std::cout<<"build time "<<timer.ElapsedMillis()<<" ms"<<std::endl;

	cudaBindTexture( NULL, BboxTex,
		builder.getLeafBoxes(),
		sizeof(Bbox3f)*size );

	h_nodes = nodes;
	h_leaves = leaves;	


	//for(int i = 0; i<h_nodes.size(); i++)
	//{ 
	//	std::cout<<" parent idx is "<<h_nodes[i].parentIdx<<" ,";
	//	
	//	if(h_nodes[i].l_isleaf)
	//	{
	//		std::cout<<i<<" left child "<<" is leaf "<<h_nodes[i].getChild(0);
	//	}
	//	else
	//	{
	//		std::cout<<i<<" left child "<<" is internal "<<h_nodes[i].getChild(0);				

	//	}
	//	if(h_nodes[i].r_isleaf)
	//	{
	//		std::cout<<" right child "<<" is leaf "<<h_nodes[i].getChild(1)<<std::endl;
	//	}
	//	else
	//	{
	//		std::cout<<" right child "<<" is internal "<<h_nodes[i].getChild(1)<<std::endl;
	//	}
	//}
	//for(int i=0; i<h_leaves.size(); i++)
	//{
	//	std::cout<<i<<" parent is "<<h_leaves[i].parentIdx<<std::endl;
	//	std::cout<<" pid is "<<h_leaves[i].pid<<std::endl;
	//}
	
	h_nodeBoxes.resize(size-1);
	Bbox3f* p_nodeBoxes = thrust::raw_pointer_cast(&h_nodeBoxes.front());
	cudaMemcpy(p_nodeBoxes,builder.getNodeBoxes(),sizeof(Bbox3f)*(size-1),cudaMemcpyDeviceToHost);

	h_leafBoxes.resize(size);
	Bbox3f* p_leafBoxes = thrust::raw_pointer_cast(&h_leafBoxes.front());
	cudaMemcpy(p_leafBoxes,builder.getLeafBoxes(),sizeof(Bbox3f)*(size),cudaMemcpyDeviceToHost);




	int frustumCount = 1;
	thrust::host_vector<pyrfrustum_t> h_frustums(frustumCount);
	for(int i=0; i< frustumCount; i++)
		h_frustums[i] = frustum;

	const uint32 BLOCK_SIZE = 128;
	int numSMs;
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
	size_t max_blocks = 65535;
	size_t n_blocks   = nih::min( max_blocks, (frustumCount + (BLOCK_SIZE*numSMs)-1) / (BLOCK_SIZE*numSMs) );
	thrust::device_vector<pyrfrustum_t> d_frustums = h_frustums;

	Bvh_Node* d_nodes = thrust::raw_pointer_cast(&(nodes.front()));
	Bvh_Node* d_leaves = thrust::raw_pointer_cast(&(leaves.front()));
	cuda::DBVH* d_bvh = NULL;
	cudaMalloc((void**)&d_bvh,sizeof(BVH));
	cudaMemcpy(d_bvh,&h_bvh,sizeof(BVH),cudaMemcpyHostToDevice);
	uint32* d_list;
	thrust::device_vector<uint32> d_vector(size*frustumCount,0);
	d_list = thrust::raw_pointer_cast(&d_vector.front());
	timer.Start();
	FrustumCullingKernel<<<n_blocks*numSMs,BLOCK_SIZE>>>(thrust::raw_pointer_cast(&d_frustums.front()),frustumCount, d_bvh, size,d_list);
	timer.Stop();
	std::cout<<"culling time "<<timer.ElapsedMillis()<<" ms"<<std::endl;
	thrust::host_vector<uint32> h_list = d_vector;
	uint32 k = 0;
	for(int i=0; i<size*frustumCount; i++)
		if(h_list[i] == 1)
			k++;
	std::cout<<"total "<<h_list.size()<<" in "<<k<<std::endl;
	
	
	
	Bintree_node* nbvh;
	uint32 nbvh_size = h_nodes.size()+h_leaves.size();
	nbvh = new Bintree_node[nbvh_size];
    Timer ctimer;
	ctimer.start();
	cuda::DFSBintree(&h_nodes,&h_leaves,&h_nodeBoxes,&h_leafBoxes,nbvh);
	//cuda::BFSBintree(&h_nodes,&h_leaves,&h_nodeBoxes,&h_leafBoxes,nbvh);
	ctimer.stop();
	std::cout<<"build DFSBintree time "<<ctimer.seconds()*1000<<"ms"<<std::endl;
	Bintree_node* d_nbvh;
	cudaMalloc((void**)&d_nbvh,sizeof(Bintree_node)*nbvh_size);
	cudaMemcpy(d_nbvh,nbvh,sizeof(Bintree_node)*nbvh_size,cudaMemcpyHostToDevice);
	cudaBindTexture( NULL, bvhTex,
		d_nbvh, sizeof(Bintree_node)*nbvh_size );

	thrust::device_vector<uint32> d_vector1(size*frustumCount,0);
	d_list = thrust::raw_pointer_cast(&d_vector1.front());
	timer.Start();
	FrustumCullingKernel<<<n_blocks*numSMs,BLOCK_SIZE>>>(thrust::raw_pointer_cast(&d_frustums.front()),frustumCount, d_nbvh, size,d_list);
	timer.Stop();
	std::cout<<"dfs bvh culling time "<<timer.ElapsedMillis()<<" ms"<<std::endl;
	h_list = d_vector1;
	k = 0;
	for(int i=0; i<size*frustumCount; i++)
		if(h_list[i] == 1)
			k++;
	std::cout<<"total "<<h_list.size()<<" in "<<k<<std::endl;
	cudaFree(d_nbvh);
	delete[] nbvh;

	Bintree_node * Nbvh;
	Nbvh = new Bintree_node[nbvh_size];   
	ctimer.start();
	cuda::BFSBintree(&h_nodes,&h_leaves,&h_nodeBoxes,&h_leafBoxes,Nbvh);
	ctimer.stop();
	std::cout<<"build BFSBintree time "<<ctimer.seconds()*1000<<"ms"<<std::endl;
	Bintree_node* d_Nbvh;
	cudaMalloc((void**)&d_Nbvh,sizeof(Bintree_node)*nbvh_size);
	cudaMemcpy(d_Nbvh,Nbvh,sizeof(Bintree_node)*nbvh_size,cudaMemcpyHostToDevice);
	thrust::device_vector<uint32> d_vector0(size*frustumCount,0);
	d_list = thrust::raw_pointer_cast(&d_vector0.front());
	timer.Start();
	FrustumCullingKernel<<<n_blocks*numSMs,BLOCK_SIZE>>>(thrust::raw_pointer_cast(&d_frustums.front()),frustumCount, d_Nbvh, size,d_list);
	timer.Stop();
	std::cout<<"BFS Nbvh culling time "<<timer.ElapsedMillis()<<" ms"<<std::endl;
	h_list = d_vector0;
	k = 0;
	for(int i=0; i<size*frustumCount; i++)
		if(h_list[i] == 1)
			k++;
	std::cout<<"total "<<h_list.size()<<" in "<<k<<std::endl;
	cudaFree(d_Nbvh);
	delete[] Nbvh;



	//Bintree h_bvhSoa;
	//cuda::DFSBintreeSOA(&h_nodes,&h_leaves,&h_nodeBoxes,&h_leafBoxes,&h_bvhSoa);
	//Bintree* d_bvhSoa;
	//cudaMalloc((void**)&d_bvhSoa,sizeof(Bintree));
	//cudaMemcpy(d_bvhSoa,&h_bvhSoa,sizeof(Bintree),cudaMemcpyHostToDevice);
	//thrust::device_vector<uint32> d_vector3(size*frustumCount,0);
	//d_list = thrust::raw_pointer_cast(&d_vector3.front());
	//timer.Start();
	//FrustumCullingKernel<<<n_blocks*numSMs,BLOCK_SIZE>>>(thrust::raw_pointer_cast(&d_frustums.front()),frustumCount, d_bvhSoa, size,d_list);
	//timer.Stop();
	//std::cout<<"dfs bvh soa culling time "<<timer.ElapsedMillis()<<" ms"<<std::endl;
	//h_list = d_vector3;
	//k = 0;
	//for(int i=0; i<size*frustumCount; i++)
	//	if(h_list[i] == 1)
	//		k++;
	//std::cout<<"total "<<h_list.size()<<" in "<<k<<std::endl;
	//cudaFree(h_bvhSoa.boxPtr);
	//cudaFree(h_bvhSoa.isLeafPtr);
	//cudaFree(h_bvhSoa.LChildPtr);
	//cudaFree(h_bvhSoa.leafEndPtr);
	//cudaFree(h_bvhSoa.leafStartPtr);
	//cudaFree(h_bvhSoa.pidPtr);
	//cudaFree(h_bvhSoa.RChildPtr);
	//cudaFree(d_bvhSoa);

	thrust::device_vector<uint32> d_vector2(size,0);
	d_list = thrust::raw_pointer_cast(&d_vector2.front());
	n_blocks   = nih::min( max_blocks, (size + (BLOCK_SIZE*numSMs)-1) / (BLOCK_SIZE*numSMs) );
	timer.Start();
	BruteforceFrustumCullingKernel<<<n_blocks*numSMs,BLOCK_SIZE>>>(thrust::raw_pointer_cast(&d_frustums.front()), builder.getLeafBoxes(), size,d_list);
	timer.Stop();
	std::cout<<"brute force culling time "<<timer.ElapsedMillis()<<" ms"<<std::endl;
	h_list = d_vector2;
	k = 0;
	for(int i=0; i<size; i++)
		if(h_list[i] == 1)
			k++;
	std::cout<<"total "<<h_list.size()<<" in "<<k<<std::endl;




	/*cuda::SimpleList<Pair> h_plist(h_points.size()*2);
	cuda::SimpleList<Pair>* d_plist=NULL;
	cudaMalloc((void**)(&d_plist),sizeof(cuda::SimpleList<Pair>));
	cudaMemcpy(d_plist,&h_plist,sizeof(cuda::SimpleList<Pair>),cudaMemcpyHostToDevice);
	const size_t blocks   = nih::min( max_blocks, (size + (BLOCK_SIZE*numSMs)-1) / (BLOCK_SIZE*numSMs) );
	timer.Start();
	CollisonDetectionKernel<<<blocks*numSMs,BLOCK_SIZE>>>(d_bvh,d_plist,size);
	timer.Stop();
	std::cout<<"collison detec time "<<timer.ElapsedMillis()<<" ms"<<std::endl;
	cudaMemcpy(&h_plist,d_plist,sizeof(cuda::SimpleList<uint32>),cudaMemcpyDeviceToHost);
	std::cout<<"overlapped "<<h_plist.size()<<std::endl;
*/
	//cudaFree(d_bvh);
}

bool BboxCompare(const Bbox3f& lbox, const Bbox3f& rbox)
{
	const double zero = 0.0001;

	return (abs(lbox.m_min[0]-rbox.m_min[0])<zero &&
		abs(lbox.m_min[1]-rbox.m_min[1])<zero && 
		abs(lbox.m_min[2]-rbox.m_min[2])<zero &&
		abs(lbox.m_max[0]-rbox.m_max[0])<zero &&
		abs(lbox.m_max[1]-rbox.m_max[1])<zero && 
		abs(lbox.m_max[2]-rbox.m_max[2])<zero );
}
struct is_frustum
{
	NIH_HOST_DEVICE bool  operator()(const TriFrustum& f)
	{
		return f.id != uint32(-1);
	}
};
struct add_frustum
{
	NIH_HOST_DEVICE TriFrustum operator()(const TriFrustum& a, const TriFrustum& b)
	{
		Bbox3f aBox(a.min,a.max);
		Bbox3f bBox(b.min,b.max);
		aBox.insert(bBox);
		TriFrustum ret;
		ret.max = aBox.m_max;
		ret.min = aBox.m_min;
		return ret;
	}
};
void GPUVirtualFrustumTest()
{
	plane_t plane;
	plane.a = 0.21;
	plane.b = 0.04;
	plane.c = -0.97;
	plane.d = -4.4;
	Vector3f p0(4.95,0.99,0.25);
	Vector3f p1(0.18,0.02,-0.98);
	Vector3f p;
	plane.intersect(p0,p1,p);

	const char* mirrorObjName = "sphere4900.obj";
	const char* sceneObjName = "sponza.obj";
	thrust::host_vector<Vector3f> h_p123Vec;	
	size_t mirrorTriNum = loadObj(mirrorObjName,h_p123Vec);
	std::cout<<h_p123Vec[0][0]<<","<<h_p123Vec[0][1]<<","<<h_p123Vec[0][2]<<std::endl;
	thrust::device_vector<Vector3f> d_p123Vec(h_p123Vec);
	Vector3f* d_p123 = thrust::raw_pointer_cast(&d_p123Vec.front());

	Vector3f eye(5,1,0);
	thrust::device_vector<Vector3f> d_eyeVec(1,eye);
	Vector3f* d_eye = thrust::raw_pointer_cast(&d_eyeVec.front());

	thrust::device_vector<TriFrustum> d_frustumVec(mirrorTriNum);
	TriFrustum* d_frustums = thrust::raw_pointer_cast(&d_frustumVec.front());

	size_t gridSize = GridSize(mirrorTriNum);

	GenerateVirFrustumKernel<<<gridSize,128>>>(d_eye,d_p123, d_frustums,150, mirrorTriNum);

	int      num_items = mirrorTriNum;     
	TriFrustum      *d_frustumsOut;             
	cudaMalloc((void**)&d_frustumsOut,sizeof(TriFrustum)*mirrorTriNum);
	int      *d_num_selected;    
	cudaMalloc((void**)&d_num_selected,sizeof(int));
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
	is_frustum op;
	cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_frustums, d_frustumsOut, d_num_selected, num_items, op);
	// Allocate temporary storage
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	// Run selection
	cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_frustums, d_frustumsOut, d_num_selected, num_items,op);
	int frustumNum;
	cudaMemcpy(&frustumNum,d_num_selected,sizeof(int),cudaMemcpyDeviceToHost);		
	cudaFree(d_temp_storage);
	std::cout<<"frustum generated "<<frustumNum<<std::endl;
	
	
	thrust::device_ptr<TriFrustum> dptr(d_frustumsOut);
	TriFrustum gFrustum = thrust::reduce(dptr,dptr+frustumNum,TriFrustum(),add_frustum());
	Bbox3f frustumBox(gFrustum.min,gFrustum.max);
	thrust::device_vector<Bbox3f> d_frustumBoxs(1,frustumBox);
	Bbox3f* d_frustumBox = thrust::raw_pointer_cast(&d_frustumBoxs.front());

	

	thrust::host_vector<Vector3f> h_points;	
	size_t size = loadObj(sceneObjName,h_points);
	thrust::device_vector<Vector3f> d_ScenePointVec(h_points);
	thrust::device_vector<Vector3f> d_SelScnPointsVec(h_points.size());
	Vector3f* d_ScenePoints = thrust::raw_pointer_cast(&d_ScenePointVec.front());
	Vector3f* d_SelScnPoints = thrust::raw_pointer_cast(&d_SelScnPointsVec.front());
	thrust::device_vector<char> d_vector2(size*3,0);
	char* d_list = thrust::raw_pointer_cast(&d_vector2.front());
	gridSize = GridSize(size);
	GpuTimer timer;
	timer.Start();
	BboxCullingKernel<<<gridSize,128>>>(d_frustumBox, d_ScenePoints, size,d_list);
	timer.Stop();
	std::cout<<"brute force culling time "<<timer.ElapsedMillis()<<" ms"<<std::endl;

	timer.Start();
	num_items = size*3;
	cudaMalloc(&d_num_selected,sizeof(int));
	// Determine temporary device storage requirements	
	temp_storage_bytes = 0;
	d_temp_storage = NULL;
	cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_ScenePoints, d_list, d_SelScnPoints, d_num_selected, num_items);
	// Allocate temporary storage
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	// Run selection
	int h_num_selected;
	cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_ScenePoints, d_list, d_SelScnPoints, d_num_selected, num_items);
	cudaMemcpy(&h_num_selected,d_num_selected,sizeof(int),cudaMemcpyDeviceToHost);
	std::cout<<"culling with gfrustum box, remaining "<<h_num_selected<<std::endl;
	cudaFree(d_num_selected);
	cudaFree(d_temp_storage);
	timer.Stop();
	std::cout<<"stream compaction time "<<timer.ElapsedMillis()<<" ms"<<std::endl;

	
	//创建bvh
	uint32 size0 = h_num_selected/3;
	size_t bvhSize = size0 +frustumNum;	
	thrust::device_vector<Vector3f> d_bvhPoints(bvhSize);
	thrust::device_vector<Bbox3f> d_bvhBoxes(bvhSize );
	Vector3f* d_points0 = thrust::raw_pointer_cast(&d_bvhPoints.front());
	Vector3f* d_points1 = thrust::raw_pointer_cast(&d_bvhPoints[size0]);
	Bbox3f* d_boxes0 = thrust::raw_pointer_cast(&d_bvhBoxes.front());
	Bbox3f* d_boxes1 = thrust::raw_pointer_cast(&d_bvhBoxes[size0]);
	gridSize = GridSize(size0);
	PrepareBvhPointKernel<<<gridSize,128>>>(d_SelScnPoints,d_points0,d_boxes0,size0);
	gridSize = GridSize(frustumNum);
	PrepareBvhPointKernel<<<gridSize,128>>>(d_frustumsOut,d_points1,d_boxes1,frustumNum);
	nih::Bbox3f h_gBox = thrust::reduce(d_bvhBoxes.begin(),d_bvhBoxes.end(),nih::Bbox3f(),Add_Bbox<nih::Vector3f>());
	thrust::device_vector<Bvh_Node> nodes(bvhSize-1);
	thrust::device_vector<Bvh_Node> leaves(bvhSize);
	cub::CachingDeviceAllocator allocator(true);
	cuda::KBvh_Builder builder(nodes,leaves,allocator);

	/*cub::DoubleBuffer<uint32> d_codes;
	size_t n_points = points.size();
	allocator.DeviceAllocate((void**)&d_codes.d_buffers[0], sizeof(uint32) * n_points);
	thrust::device_ptr<uint32> d_ptr(d_codes.d_buffers[0]);
	thrust::transform(
	d_points.begin(),
	d_points.begin() + n_points,
	d_ptr,
	morton_functor<uint32>( gBox ) );
	uint32 * h_codes = (uint32*)malloc(sizeof(uint32)*n_points);
	cudaMemcpy(h_codes,d_codes.d_buffers[0],sizeof(uint32)*n_points,cudaMemcpyDeviceToHost);
	for(int i=0; i<n_points; i++)
	std::cout<<h_codes[i]<<std::endl;*/
	cuda::DBVH h_bvh;
	/*cudaMalloc((void**)&d_bvh,sizeof(cuda::DBVH));*/

	timer.Start();
	builder.build(h_gBox,d_bvhPoints.begin(),d_bvhPoints.end(),d_bvhBoxes.begin(),d_bvhBoxes.end(),&h_bvh);
	timer.Stop();
	std::cout<<"build time "<<timer.ElapsedMillis()<<" ms"<<std::endl;

	/*thrust::host_vector<Bvh_Node> h_nodes = nodes;
	thrust::host_vector<Bvh_Node> h_leaves = leaves;*/	
	//for(int i = 0; i<h_nodes.size(); i++)
	//{ 
	//	std::cout<<" parent idx is "<<h_nodes[i].parentIdx<<" ,";
	//	
	//	if(h_nodes[i].l_isleaf)
	//	{
	//		std::cout<<i<<" left child "<<" is leaf "<<h_nodes[i].getChild(0);
	//	}
	//	else
	//	{
	//		std::cout<<i<<" left child "<<" is internal "<<h_nodes[i].getChild(0);				

	//	}
	//	if(h_nodes[i].r_isleaf)
	//	{
	//		std::cout<<" right child "<<" is leaf "<<h_nodes[i].getChild(1)<<std::endl;
	//	}
	//	else
	//	{
	//		std::cout<<" right child "<<" is internal "<<h_nodes[i].getChild(1)<<std::endl;
	//	}
	//}
	/*for(int i=0; i<h_leaves.size(); i++)
	{
		std::cout<<i<<" parent is "<<h_leaves[i].parentIdx<<std::endl;
		std::cout<<" pid is "<<h_leaves[i].pid<<std::endl;
	}*/
	timer.Start();
	cuda::DBVH* d_bvh = NULL;
	cudaMalloc((void**)&d_bvh,sizeof(BVH));
	cudaMemcpy(d_bvh,&h_bvh,sizeof(BVH),cudaMemcpyHostToDevice);

	gridSize = GridSize(bvhSize);
	thrust::device_vector<uint32> d_cullingResult(frustumNum*size0);
	uint32* d_clist = thrust::raw_pointer_cast(&d_cullingResult.front());
	//FrustumCullingKernel<<<gridSize,128>>>(d_frustumsOut,frustumNum,d_bvh, size0,d_clist);
	FrustumCullingKernel<<<gridSize,128>>>(d_frustumsOut,d_bvh,frustumNum, size0,d_clist);
	timer.Stop();
	std::cout<<"culling time "<<timer.ElapsedMillis()<<" ms"<<std::endl;
}
int main(int argc, char** argv)
{
	GPUVirtualFrustumTest();
	return;
	char* fileName ="testbox.obj";
	if (argc == 2)
	{
		fileName = (argv[1]);
	}
	// Projection matrix : 45° Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
	glm::mat4 Projection = glm::perspective(45.0f, 4.0f / 3.0f, 0.01f, 50.0f);
	// Camera matrix
	glm::mat4 View       = glm::lookAt(
		glm::vec3(0,0,-7), // Camera is at (4,3,3), in World Space
		glm::vec3(0,0,0), // and looks at the origin
		glm::vec3(0,1,0)  // Head is up (set to 0,-1,0 to look upside-down)
		);
	// Model matrix : an identity matrix (model will be at the origin)
	glm::mat4 Model      = glm::mat4(1.0f);  // Changes for each model !

	// Our ModelViewProjection : multiplication of our 3 matrices
	glm::mat4 MVP        = Projection * View * Model; // Remember, matrix multiplication is the other way around

	Matrix4x4 mvp;		
	memcpy(&mvp,&MVP[0][0],16*sizeof(float));
	pyrfrustum_t frustum;
	ExtractPlanesGL(frustum.planes,mvp,true);

	thrust::host_vector<Vector3f> h_points;
	thrust::host_vector<Bbox3f> h_boxes;
	Bbox3f gBox;
	loadObj(fileName,h_points,h_boxes,gBox);
	//loadRandom(3650,h_points,h_boxes,gBox);

	thrust::host_vector<Bvh_Node> h_nodes,h_leaves;
	thrust::host_vector<Bbox3f> h_nodeBoxes,h_leafBoxes;
	std::cout<<"gpu:"<<std::endl;
	GpuKlbvhTest(h_points,h_boxes,gBox,frustum,h_nodes,h_leaves,h_nodeBoxes,h_leafBoxes);
	std::cout<<"cpu:"<<std::endl;
	BVH cpuBvh;
	CpuKlbvhTest(h_points,h_boxes,gBox,frustum,cpuBvh);

	std::cout<<"比较"<<std::endl;
	if (cpuBvh.nodes.size() != h_nodes.size())
	{
		std::cout<<"size is different!"<<std::endl;
	}
	for (int i=0; i<cpuBvh.nodes.size(); i++)
	{
		if (!cpuBvh.nodes[i].equal(h_nodes[i]))
		{
			std::cout<<"node "<<i<<"is different"<<std::endl;
			break;
		}
		if (cpuBvh.leafs[i].parentIdx != h_leaves[i].parentIdx)
		{
			std::cout<<"leaf "<<i<<"is different"<<std::endl;
			std::cout<<"cpu "<<cpuBvh.leafs[i].parentIdx<<" gpu "<<h_leaves[i].parentIdx<<std::endl;
			break;
		}
		if (!BboxCompare(cpuBvh.node_Boxes[i],h_nodeBoxes[i]))
		{
			std::cout<<"node box "<<i<<"is different"<<std::endl;
			break;
		}
		if (!BboxCompare(cpuBvh.leaf_Boxes[i],h_leafBoxes[i]))
		{
			std::cout<<"leaf box "<<i<<"is different"<<std::endl;
			std::cout<<"cpu ";
			printBbox3f(cpuBvh.leaf_Boxes[i]);
			std::cout<<"gpu ";
			printBbox3f(h_leafBoxes[i]);
			break;
		}
	}

	return 0;
}
