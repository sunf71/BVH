#pragma once
#include "types.h"
#include "bbox.h"

using namespace nih;
struct Bvh_Node
{
	NIH_HOST_DEVICE Bvh_Node():parentIdx(-1){};
	NIH_HOST_DEVICE Bvh_Node(uint32 cIdx):childIdx(cIdx),parentIdx(-1){};
	//2 child,0 or 1
	NIH_HOST_DEVICE uint32 getChild(int i)
	{
		return childIdx + i;
	}
	NIH_HOST_DEVICE bool equal(const Bvh_Node& node)
	{
		return (node.childIdx == childIdx) &&
			(node.id == id) && node.leafEnd == leafEnd && 
			node.leafStart == leafStart && node.l_isleaf == l_isleaf &&
			node.parentIdx == parentIdx && node.r_isleaf == r_isleaf;
	}
	bool l_isleaf;
	bool r_isleaf;
	uint32	childIdx;	// child index
	uint32 parentIdx; //parent index
	uint32 id;
	uint32 leafStart;//包含的叶子节点开始
	uint32 leafEnd;//包含的叶子节点结束
	uint32 pid;//primitive id
	bool isLeaf;//是否是叶子节点
	uint32 nid;//新树的id
};

///
/// A middle-split binary tree node.
/// A node can either be a leaf and have no children, or be
/// an internal split node. If a split node, it can either
/// have one or two children: for example, it can have one
/// if a set of points is concentrated in one half-space.
///
struct Bintree_node
{
	NIH_HOST_DEVICE Bintree_node()
	{
		lChild = RChild = 0;
	}
	NIH_HOST_DEVICE bool isLeaf()
	{
		return leafStart == leafEnd;
	}
    //Bbox3f box;
	uint32 minX;
	uint32 minY;
	uint32 minZ;
	uint32 maxX;
	uint32 maxY;
	uint32 maxZ;
	uint32 lChild;
	uint32 RChild;	
	uint32 pid;
	uint32 leafStart;
	uint32 leafEnd;
};
struct Bintree_Node
{
	NIH_HOST_DEVICE bool isLeaf()
	{
		return leafStart ==  leafEnd;
	}
    uint32 lChild;
	uint32 RChild;
	Bbox3f lBox;
	Bbox3f rBox;
	uint32 leafStart;
	uint32 leafEnd;
};

struct Bintree
{
	uint32* LChildPtr;
	uint32* RChildPtr;
	bool* isLeafPtr;
	Bbox3f* boxPtr;
	uint32* pidPtr;
	uint32* leafStartPtr;
	uint32* leafEndPtr;

};

NIH_HOST_DEVICE  inline uint32       floatToBits     (float a)         { return *(uint32*)&a; }
NIH_HOST_DEVICE inline float           bitsToFloat     (uint32 a)         { return *(float*)&a; }
NIH_HOST_DEVICE inline void  SetBox(Bintree_node* node, Bbox3f* box)
{
	node->minX = floatToBits(box->m_min[0]);
	node->minY = floatToBits(box->m_min[1]);
	node->minZ = floatToBits(box->m_min[2]);
	node->maxX = floatToBits(box->m_max[0]);
	node->maxY = floatToBits(box->m_max[1]);
	node->maxZ = floatToBits(box->m_max[2]);
}