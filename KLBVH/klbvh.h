#pragma once
#include <vector>
#include "types.h"
#include "bvh.h"
#include <queue>
#include "Bbox.h"
#include <thrust\host_vector.h>
#include <iostream>
#define u 99
#define sign(x) (( x >> 31 ) | ( (unsigned int)( -x ) >> 31 ))
using namespace nih;
using namespace std;
extern int globalCounter;
__inline int nlz(unsigned x) {

	static char table[64] =
	{32,20,19, u, u,18, u, 7,  10,17, u, u,14, u, 6, u,
	u, 9, u,16, u, u, 1,26,   u,13, u, u,24, 5, u, u,
	u,21, u, 8,11, u,15, u,   u, u, u, 2,27, 0,25, u,
	22, u,12, u, u, 3,28, u,  23, u, 4,29, u, u,30,31};

	x = x | (x >> 1);    // Propagate leftmost
	x = x | (x >> 2);    // 1-bit to the right.
	x = x | (x >> 4);
	x = x | (x >> 8);
	x = x & ~(x >> 16);
	x = x*0xFD7049FF;    // Activate this line or the following 3.
	// x = (x << 9) - x;    // Multiply by 511.
	// x = (x << 11) - x;   // Multiply by 2047.
	// x = (x << 14) - x;   // Multiply by 16383.
	return table[x >> 26];
}

void inline printNode(Bvh_Node* node)
{
	globalCounter++;
	//if (node!=NULL)
	//{
	//	cout<<" parent idx is "<<node->parentIdx<<" ";
	//	if (node->l_isleaf)
	//	cout<<" left child is leaf  ";
	//	if (node->r_isleaf)
	//	cout<<" right child is leaf "<<endl;
	//	//cout<<" child index "<<node->getChild(0)<<endl;
	//}
	//else
	//	cout<<"invalid node"<<endl;
	if (node->isLeaf)
		cout<<"leaf node"<<endl;
	else
		cout<<node->id<<endl;
}
struct BVH
{
	vector<Bvh_Node> nodes;
	vector<Bbox3f> node_Boxes;
	vector<Bvh_Node> leafs;
	vector<Bbox3f> leaf_Boxes;
	vector<uint32> indices;
	void print()
	{
		for(int i = 0; i<nodes.size(); i++)
		{ 
			cout<<" parent idx is "<<nodes[i].parentIdx<<" ";
			if(nodes[i].l_isleaf)
			{
				cout<<i<<" left child "<<" is leaf "<<nodes[i].getChild(0);
			}
			else
			{
				cout<<i<<" left child "<<" is internal "<<nodes[i].getChild(0);				

			}
			if(nodes[i].r_isleaf)
			{
				cout<<" right child "<<" is leaf "<<nodes[i].getChild(1)<<endl;
			}
			else
			{
				cout<<" right child "<<" is internal "<<nodes[i].getChild(1)<<endl;
			}
		}
		for(int i=0; i<leafs.size(); i++)
		{
			cout<<i<<" parent is "<<leafs[i].parentIdx<<endl;
		}
	}

	void traversal(void func(Bvh_Node* node))
	{
		const uint32 stack_size  = 32;
		Bvh_Node stack[stack_size];
		uint32 top = 0;
		stack[top++] = nodes[0];
		while(top>0)
		{
			Bvh_Node node = stack[--top];

			func(&node);
			if(node.r_isleaf)
				func(&leafs[node.getChild(1)]);
			else
				stack[top++] = nodes[node.getChild(1)];
			if (node.l_isleaf)
				func(&leafs[node.getChild(0)]);
			else
				stack[top++] = nodes[node.getChild(0)];
		}
	}
};

struct Primitive
{
	Primitive(Vector3f& c):center(c)
	{
		bbox.insert(c);
	}
	Primitive(Vector3f& c, Bbox3f& b):center(c),bbox(b){}
	Primitive(){}
	bool operator()( const Primitive& lhs, const Primitive& rhs) const
	{
		return lhs.code<rhs.code;
	}
	Vector3f center;
	Bbox<Vector3f> bbox;
	uint32 code;
};
typedef vector<Primitive>::const_iterator PrimitiveIterator;
typedef vector<Bbox<Vector3f>>::const_iterator BboxIterator;
class Bvh_Builder
{
public:
	Bvh_Builder(){}
	void build(const Bbox<Vector3f> globalBbox,PrimitiveIterator begin, PrimitiveIterator end, BVH& bvh);
	void assignAABBs(BVH& bvh);
	//
	int theta(uint32 i, uint32 j)
	{
		if (j>m_size-1)
			return -1;
		uint32 a1 = m_keys[i];
		uint32 a2 = m_keys[j];
		if ( a1==a2)
		{ 
			return nlz(i^j) + 32;
		}
		uint32 a = a1^a2;

		return nlz(a); 

	}
protected:
	vector<uint32> m_keys;	

	uint32 m_size;




};

inline void DFSBinTree(BVH& bvh, Bintree_node* nbvh)
{

	uint32 offset = 0;
	const uint32 stack_size  = 64;
	Bvh_Node* stack[stack_size];
	uint32 top = 0;
	bvh.nodes[0].nid = 0;
	Bvh_Node* node = &bvh.nodes[0];
	while(top >0 || node != NULL)
	{
		while(node != NULL)
		{
			//cout<<node->id<<endl;
			//nbvh[offset].box = bvh.node_Boxes[node->id];			
			SetBox(&nbvh[offset],&bvh.node_Boxes[node->id]);
			nbvh[offset].leafStart = node->leafStart;
			nbvh[offset].leafEnd = node->leafEnd;
			nbvh[offset].pid = node->id;
			nbvh[offset].lChild = offset+1;
			node->nid = offset;
			offset++;

			stack[top++] = node;
			if (node->l_isleaf)
			{
				SetBox(&nbvh[offset],&bvh.leaf_Boxes[bvh.indices[node->getChild(0)]]);
				//nbvh[offset].box = bvh.leaf_Boxes[bvh.indices[node->getChild(0)]];
				nbvh[offset].leafStart = nbvh[offset].leafEnd = nbvh[offset].pid =node->getChild(0);
				
				offset++;
				//cout<<"leaf "<<node->getChild(0)<<endl;
				node = NULL;
			}
			else
			{
				node = &bvh.nodes[node->getChild(0)];
			}
		}
		node = stack[--top];
		nbvh[node->nid].RChild = offset;
		if (node->r_isleaf)
		{			
			SetBox(&nbvh[offset],&bvh.leaf_Boxes[bvh.indices[node->getChild(1)]]);
			nbvh[offset].pid = node->getChild(1);
			nbvh[offset].leafStart = nbvh[offset].leafEnd = nbvh[offset].pid =node->getChild(1);
			offset++;
			//cout<<"leaf "<<node->getChild(1)<<endl;
			node = NULL;
		}
		else
			node =  &bvh.nodes[node->getChild(1)];
	}
}

inline void BFSBinTree(BVH& bvh, Bintree_node* nbvh)
{
	const uint32 size = 65535;
	Bvh_Node* queue[size+1];
	bvh.nodes[0].nid = 0;
	bvh.nodes[0].parentIdx = 0;
	uint32 rear,front;
	rear = front = 0;
	queue[rear++] = &bvh.nodes[0];
	uint32 offset = 0;
	while(front != rear)
	{
		Bvh_Node* p = queue[front];
		front = (front +1)&(size);
		p->nid = offset;
		if (p->isLeaf)
		{
			
			nbvh[offset].leafStart = nbvh[offset].leafEnd = nbvh[offset].pid = bvh.indices[p->id];
			//nbvh[offset].box = bvh.leaf_Boxes[nbvh[offset].pid];
			SetBox(&nbvh[offset],&bvh.leaf_Boxes[nbvh[offset].pid]);
			//std::cout<<"leaf "<<nbvh[offset].pid<<std::endl;
		}
		else
		{
			SetBox(&nbvh[offset],&bvh.node_Boxes[p->id]);
			//nbvh[offset].box = bvh.node_Boxes[p->id];
			nbvh[offset].leafStart = p->leafStart;
			nbvh[offset].leafEnd = p->leafEnd;
			//std::cout<<"node "<<p->id<<std::endl;
			if (p->l_isleaf)
			{
				queue[rear] = &bvh.leafs[p->getChild(0)];
			}
			else
			{
				queue[rear] = &bvh.nodes[p->getChild(0)];
			}
			rear = (rear+1)&(size);
			if (p->r_isleaf)
			{
				queue[rear] = &bvh.leafs[p->getChild(1)];
			}
			else
			{
				queue[rear] = &bvh.nodes[p->getChild(1)];
			}
			rear = (rear+1)&(size);
		}
		uint32 ptr = bvh.nodes[p->parentIdx].nid;
		if (nbvh[ptr].lChild == 0)
			nbvh[ptr].lChild = offset;
		else
			nbvh[ptr].RChild = offset;
		offset++;



	}

}