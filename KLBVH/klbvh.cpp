#include "klbvh.h"
#include "MortonCode.h"
#include <algorithm>
#include <bitset>

// This function returns true if the first pair is "less"
// than the second one according to some metric
// In this case, we say the first pair is "less" if the first element of the first pair
// is less than the first element of the second pair
bool pairCompare(const std::pair<uint32, Bbox3f>& firstElem, const std::pair<uint32, Bbox3f>& secondElem) {
  return firstElem.first < secondElem.first;

}
void Bvh_Builder::build(const Bbox<Vector3f> bbox,PrimitiveIterator begin, PrimitiveIterator end, BVH& bvh)
{
	//std::vector<std::pair<uint32,Bbox3f>> map;
	std::vector<std::pair<uint32,uint32>> indexMap;
	morton_functor<uint32> morton(bbox);
	uint32 t=0;
	//assign morton code
	for( PrimitiveIterator itr = begin; itr!= end; itr++)
	{
		/*m_keys.push_back(morton(itr->center));
		m_leaveboxes.push_back(itr->bbox);*/
		uint32 key = morton(itr->center);
		/*map.push_back(std::pair<uint32,Bbox3f>(key,itr->bbox));*/
		indexMap.push_back(std::pair<uint32,uint32>(key,t++));
		bvh.leaf_Boxes.push_back(itr->bbox);
	}
	/*m_keys.push_back(1);
	m_keys.push_back(2);
	m_keys.push_back(4);
	m_keys.push_back(5);
	m_keys.push_back(19);
	m_keys.push_back(24);
	m_keys.push_back(25);
	m_keys.push_back(30);
	m_keys.push_back(35);
	m_keys.push_back(38);
	m_keys.push_back(42);
	m_keys.push_back(45);*/

	//sort
	//std::sort(m_keys.begin(),m_keys.end());
	/*std::sort(map.begin(),map.end(),pairCompare);*/
	std::sort(indexMap.begin(),indexMap.end());
	/*for( std::vector<std::pair<uint32,Bbox3f>>::const_iterator itr= map.begin(); itr!= map.end(); itr++)
	{
		m_keys.push_back(itr->first);
		bvh.leaf_Boxes.push_back(itr->second);
		
	}*/
	for( std::vector<std::pair<uint32,uint32>>::const_iterator itr= indexMap.begin(); itr!= indexMap.end(); itr++)
	{
		m_keys.push_back(itr->first);
		bvh.indices.push_back(itr->second);
		
	}

	/*std::cout<<"sorted keys"<<std::endl;
	for(int i=0; i<m_keys.size(); i++)
	{
		std::cout<<m_keys[i]<<" ";
		std::bitset<32> bitvec((int)m_keys[i]);
		std::cout<<bitvec<<std::endl;;
	}*/

	/*std::cout<<"sorted leaf boxes"<<std::endl;
	for(int i=0; i<bvh.leaf_Boxes.size(); i++)
		printBbox3f(bvh.leaf_Boxes[i]);*/

	m_size = end - begin;

	bvh.nodes.resize(m_size-1);
	bvh.leafs.resize(m_size);

	for(uint32 i=0; i<m_size-1; i++)
	{
		int d = sign(theta(i,i+1)-theta(i,i-1));
		int tMin = theta(i,i-d);
		uint32 lMax = 2;
		while(theta(i,i+lMax*d) > tMin)
			lMax *= 2;

		uint32 l = 0;
		for(uint32 t = lMax/2; t>=1; t/=2)
		{
			if (theta(i,i+(t+l)*d) > tMin)
				l += t;
		}
		uint32 j = i + l*d;
		uint32 tNode = theta(i,j);
		uint32 s = 0;
		for(uint32 t = ceil(l/2.0); t!=1; t=ceil(t/2.0))
		{
			if (theta(i,i+(s+t)*d) > tNode)
				s = s+t;
		}
		if (theta(i,i+(s+1)*d) > tNode)
				s = s+1;
		uint32 gama  = i + s*d + min(d,0);
		bvh.nodes[i].childIdx = gama;
		bvh.nodes[i].id = i;
		bvh.nodes[i].isLeaf = false;
		bvh.nodes[i].leafStart = min(i,j);
		bvh.nodes[i].leafEnd  = max(i,j);
		if(min(i,j) == gama) 
		{
			bvh.nodes[i].l_isleaf = true; 		
			bvh.leafs[gama].parentIdx = i;
			bvh.leafs[gama].isLeaf = true;
			bvh.leafs[gama].id = gama;
		}
		else
		{
			bvh.nodes[i].l_isleaf = false; 	
			bvh.nodes[gama].parentIdx = i;
		}
		if (max(i,j) == gama +1)
		{
			bvh.nodes[i].r_isleaf = true;	
			bvh.leafs[gama+1].parentIdx = i;
			bvh.leafs[gama+1].isLeaf = true;
			bvh.leafs[gama+1].id = gama+1;
		}
		else
		{
			bvh.nodes[i].r_isleaf = false;
			bvh.nodes[gama+1].parentIdx = i;
		}
	}

	assignAABBs(bvh);

	/*std::cout<<"node boxes"<<std::endl;
	for(int i =0; i< bvh.node_Boxes.size(); i++)
		printBbox3f(bvh.node_Boxes[i]);*/


}

void Bvh_Builder::assignAABBs(BVH& bvh)
{
	vector<uint32> counters(bvh.nodes.size(),0);
	bvh.node_Boxes.resize(bvh.nodes.size());
	for(int i=0; i<bvh.leafs.size(); i++)
	{
		int j = bvh.leafs[i].parentIdx;
		while (j>=0)
		{
			if (counters[j] ==0)
			{
				counters[j]++;
			}
			else if(counters[j] == 1)
			{
				Bbox3f lBox, rBox;
				Bvh_Node node = bvh.nodes[j];
				int lId = node.getChild(0);
				int rId = node.getChild(1);
				if (node.l_isleaf) 
					lBox = bvh.leaf_Boxes[bvh.indices[lId]];
				else
					lBox = bvh.node_Boxes[lId];
				if (node.r_isleaf)
					rBox = bvh.leaf_Boxes[bvh.indices[rId]];
				else
					rBox = bvh.node_Boxes[rId];

				lBox.insert(rBox);
				bvh.node_Boxes[j] = lBox;

				j = node.parentIdx;
			}
		}
	}

}