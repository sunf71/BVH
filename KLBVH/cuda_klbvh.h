#pragma once
#include "types.h"
#include "bvh.h"
#include "bbox.h"
#include <queue>
#include <thrust/device_vector.h>

#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>


namespace cuda
{

	struct Primitives
	{
		cub::DoubleBuffer<nih::Bbox3f> boxes;
		cub::DoubleBuffer<nih::uint32> indices;
		cub::DoubleBuffer<nih::uint32> codes;
	};

	struct DBVH
	{
		Bvh_Node* nodes;	
		Bbox3f* nodeBoxes;
		Bbox3f* leafBoxes;
		Bvh_Node* leaves;
		NIH_HOST_DEVICE Bvh_Node* getRoot()
		{
			return nodes;
		}
		NIH_HOST_DEVICE Bbox3f getNodeBox(Bvh_Node* node)
		{
			return nodeBoxes[node->id];

		}
		NIH_HOST_DEVICE Bvh_Node* getLChild(Bvh_Node* node)
		{
			return &nodes[node->getChild(0)];
		}
		NIH_HOST_DEVICE Bvh_Node* getRChild(Bvh_Node* node)
		{
			return &nodes[node->getChild(1)];
		}
		NIH_HOST_DEVICE Bvh_Node* getRLeafChild(Bvh_Node* node)
		{
			return &leaves[node->getChild(1)];
		}
		NIH_HOST_DEVICE Bvh_Node* getLLeafChild(Bvh_Node* node)
		{
			return &leaves[node->getChild(0)];
		}
		NIH_HOST_DEVICE Bbox3f getLeafBox(Bvh_Node* node)
		{
			return leafBoxes[node->pid];
		}
	};
	void BFSBintree(thrust::host_vector<Bvh_Node>* h_nodes,
		thrust::host_vector<Bvh_Node>* h_leaves,
		thrust::host_vector<Bbox3f>* h_nodeBoxes,
		thrust::host_vector<Bbox3f>* h_leafBoxes,
		Bintree_node*  nbvh)
	{
		Bvh_Node* nodes = &(*h_nodes)[0];
		Bvh_Node* leaves = &(*h_leaves)[0];
		Bbox3f* nodeBoxes = &(*h_nodeBoxes)[0];
		Bbox3f* leafBoxes = &(*h_leafBoxes)[0];
		const uint32 size = 65535;
		Bvh_Node* queue[size+1];
		nodes[0].nid = 0;
		nodes[0].parentIdx = 0;
		uint32 rear,front;
		rear = front = 0;
		queue[rear++] = &nodes[0];
		uint32 offset = 0;
		while(front != rear)
		{
			Bvh_Node* p = queue[front];
			front = (front +1)&(size);


			p->nid = offset;
			if (p->isLeaf)
			{

				nbvh[offset].leafStart = nbvh[offset].leafEnd = nbvh[offset].pid = p->id;
				SetBox(&nbvh[offset],&leafBoxes[p->pid]);

				//std::cout<<"leaf "<<nbvh[offset].pid<<std::endl;
			}
			else
			{
				SetBox(&nbvh[offset],&nodeBoxes[p->id]);

				nbvh[offset].leafStart = p->leafStart;
				nbvh[offset].leafEnd = p->leafEnd;
				//std::cout<<"node "<<p->id<<std::endl;
				if (p->l_isleaf)
				{
					queue[rear] = &leaves[p->getChild(0)];
				}
				else
				{
					queue[rear] = &nodes[p->getChild(0)];
				}
				rear = (rear+1)&(size);
				if (p->r_isleaf)
				{
					queue[rear] = &leaves[p->getChild(1)];
				}
				else
				{
					queue[rear] = &nodes[p->getChild(1)];
				}
				rear = (rear+1)&(size);
			}
			uint32 ptr = nodes[p->parentIdx].nid;
			if (nbvh[ptr].lChild == 0)
				nbvh[ptr].lChild = offset;
			else
				nbvh[ptr].RChild = offset;
			offset++;



		}
	}
	void DFSBintree(thrust::host_vector<Bvh_Node>* h_nodes,
		thrust::host_vector<Bvh_Node>* h_leaves,
		thrust::host_vector<Bbox3f>* h_nodeBoxes,
		thrust::host_vector<Bbox3f>* h_leafBoxes,
		Bintree_node*  nbvh)
	{
		Bvh_Node* nodes = &(*h_nodes)[0];
		Bvh_Node* leaves = &(*h_leaves)[0];
		Bbox3f* nodeBoxes = &(*h_nodeBoxes)[0];
		Bbox3f* leafBoxes = &(*h_leafBoxes)[0];
		uint32 offset = 0;
		const uint32 stack_size  = 64;
		Bvh_Node* stack[stack_size];
		uint32 top = 0;
		nodes[0].nid = 0;
		Bvh_Node* node = &nodes[0];
		while(top >0 || node != NULL)
		{
			while(node != NULL)
			{
				//cout<<node->id<<endl;
				SetBox(&nbvh[offset],&nodeBoxes[node->id]);
				//nbvh[offset].box = nodeBoxes[node->id];			
				nbvh[offset].leafStart = node->leafStart;
				nbvh[offset].leafEnd = node->leafEnd;
				nbvh[offset].pid = node->id;
				nbvh[offset].lChild = offset+1;
				node->nid = offset;
				offset++;

				stack[top++] = node;
				if (node->l_isleaf)
				{

					nbvh[offset].leafStart = nbvh[offset].leafEnd = nbvh[offset].pid = node->getChild(0);
					SetBox(&nbvh[offset],&leafBoxes[leaves[node->getChild(0)].pid]);

					offset++;
					//cout<<"leaf "<<node->getChild(0)<<endl;
					node = NULL;
				}
				else
				{
					node = &nodes[node->getChild(0)];
				}
			}
			node = stack[--top];
			nbvh[node->nid].RChild = offset;
			if (node->r_isleaf)
			{			
				SetBox(&nbvh[offset],&leafBoxes[leaves[node->getChild(1)].pid]);
				nbvh[offset].leafStart = nbvh[offset].leafEnd = nbvh[offset].pid = node->getChild(1);
				offset++;
				//cout<<"leaf "<<node->getChild(1)<<endl;
				node = NULL;
			}
			else
				node =  &nodes[node->getChild(1)];
		}

	}

	void DFSBintree(thrust::host_vector<Bvh_Node>* h_nodes,
		thrust::host_vector<Bvh_Node>* h_leaves,
		thrust::host_vector<Bbox3f>* h_nodeBoxes,
		thrust::host_vector<Bbox3f>* h_leafBoxes,
		Bintree_Node*  nbvh)
	{
		Bvh_Node* nodes = &(*h_nodes)[0];
		Bvh_Node* leaves = &(*h_leaves)[0];
		Bbox3f* nodeBoxes = &(*h_nodeBoxes)[0];
		Bbox3f* leafBoxes = &(*h_leafBoxes)[0];
		uint32 offset = 0;
		const uint32 stack_size  = 64;
		Bvh_Node* stack[stack_size];
		uint32 top = 0;
		nodes[0].nid = 0;
		Bvh_Node* node = &nodes[0];
		while(top >0 || node != NULL)
		{
			while(node != NULL)
			{
				//cout<<node->id<<endl;

				nbvh[offset].leafStart = node->leafStart;
				nbvh[offset].leafEnd = node->leafEnd;			
				nbvh[offset].lChild = offset+1;
				node->nid = offset;
				offset++;

				stack[top++] = node;
				if (node->l_isleaf)
				{
					nbvh[offset].leafStart =  nbvh[offset].leafEnd = leaves[node->getChild(0)].pid;
					nbvh[offset-1].lBox = nbvh[offset].lBox = nbvh[offset].rBox = leafBoxes[nbvh[offset].leafStart];				
					offset++;
					//cout<<"leaf "<<node->getChild(0)<<endl;
					node = NULL;
				}
				else
				{
					nbvh[offset-1].lBox = nodeBoxes[node->getChild(0)];
					node = &nodes[node->getChild(0)];
				}
			}
			node = stack[--top];
			nbvh[node->nid].RChild = offset;

			if (node->r_isleaf)
			{			
				nbvh[node->nid].rBox = nbvh[offset].lBox = nbvh[offset].rBox= leafBoxes[leaves[node->getChild(1)].pid];
				nbvh[offset].leafStart= nbvh[offset].leafEnd = leaves[node->getChild(1)].pid;

				offset++;
				//cout<<"leaf "<<node->getChild(1)<<endl;
				node = NULL;
			}
			else
			{
				node =  &nodes[node->getChild(1)];
				nbvh[node->nid].rBox = nodeBoxes[node->id];
			}
		}

	}

	void DFSBintreeSOA(thrust::host_vector<Bvh_Node>* h_nodes,
		thrust::host_vector<Bvh_Node>* h_leaves,
		thrust::host_vector<Bbox3f>* h_nodeBoxes,
		thrust::host_vector<Bbox3f>* h_leafBoxes,
		Bintree*  dbvh)
	{

		Bintree bvh;
		Bintree* nbvh = &bvh;
		uint32 size = (*h_nodes).size()+(*h_leaves).size();
		nbvh->boxPtr = new Bbox3f[size];
		nbvh->isLeafPtr = new bool[size];
		nbvh->LChildPtr = new uint32[size];
		nbvh->leafEndPtr = new uint32[size];
		nbvh->leafStartPtr = new uint32[size];
		nbvh->pidPtr = new uint32[size];
		nbvh->RChildPtr = new uint32[size];

		Bvh_Node* nodes = &(*h_nodes)[0];
		Bvh_Node* leaves = &(*h_leaves)[0];
		Bbox3f* nodeBoxes = &(*h_nodeBoxes)[0];
		Bbox3f* leafBoxes = &(*h_leafBoxes)[0];
		uint32 offset = 0;
		const uint32 stack_size  = 64;
		Bvh_Node* stack[stack_size];
		uint32 top = 0;
		nodes[0].nid = 0;
		Bvh_Node* node = &nodes[0];
		while(top >0 || node != NULL)
		{
			while(node != NULL)
			{
				//cout<<node->id<<endl;
				nbvh->boxPtr[offset] = nodeBoxes[node->id];
				nbvh->isLeafPtr[offset] = false;
				nbvh->leafStartPtr[offset] = node->leafStart;
				nbvh->leafEndPtr[offset] = node->leafEnd;
				nbvh->pidPtr[offset] = node->id;
				nbvh->LChildPtr[offset] = offset+1;
				node->nid = offset;
				offset++;

				stack[top++] = node;
				if (node->l_isleaf)
				{

					nbvh->isLeafPtr[offset] = true;
					nbvh->pidPtr[offset] = leaves[node->getChild(0)].pid;
					nbvh->boxPtr[offset] = leafBoxes[nbvh->pidPtr[offset]];
					offset++;
					//cout<<"leaf "<<node->getChild(0)<<endl;
					node = NULL;
				}
				else
				{
					node = &nodes[node->getChild(0)];
				}
			}
			node = stack[--top];
			nbvh->RChildPtr[node->nid] = offset;
			if (node->r_isleaf)
			{			
				nbvh->isLeafPtr[offset] = true;
				nbvh->pidPtr[offset] = leaves[node->getChild(1)].pid;
				nbvh->boxPtr[offset] = leafBoxes[nbvh->pidPtr[offset]];
				offset++;
				//cout<<"leaf "<<node->getChild(1)<<endl;
				node = NULL;
			}
			else
				node =  &nodes[node->getChild(1)];
		}


		cudaMalloc((void**)&(dbvh->boxPtr),sizeof(Bbox3f)*size);
		cudaMemcpy(dbvh->boxPtr,nbvh->boxPtr,sizeof(Bbox3f)*size,cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dbvh->isLeafPtr,sizeof(bool)*size);
		cudaMemcpy(dbvh->isLeafPtr,nbvh->isLeafPtr,sizeof(bool)*size,cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dbvh->LChildPtr,sizeof(uint32)*size);
		cudaMemcpy(dbvh->LChildPtr,nbvh->LChildPtr,sizeof(uint32)*size,cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dbvh->leafEndPtr,sizeof(uint32)*size);
		cudaMemcpy(dbvh->leafEndPtr,nbvh->leafEndPtr,sizeof(uint32)*size,cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dbvh->leafStartPtr,sizeof(uint32)*size);
		cudaMemcpy(dbvh->leafStartPtr,nbvh->leafStartPtr,sizeof(uint32)*size,cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dbvh->pidPtr,sizeof(uint32)*size);
		cudaMemcpy(dbvh->pidPtr,nbvh->pidPtr,sizeof(uint32)*size,cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dbvh->RChildPtr,sizeof(uint32)*size);
		cudaMemcpy(dbvh->RChildPtr,nbvh->RChildPtr,sizeof(uint32)*size,cudaMemcpyHostToDevice);



	}

	class KBvh_Builder
	{
	public:
		KBvh_Builder(thrust::device_vector<Bvh_Node>& nodes,
			thrust::device_vector<Bvh_Node>& leaves,
			cub::CachingDeviceAllocator& allocator);
		~KBvh_Builder()
		{
			m_allocator->DeviceFree(m_primitives.codes.d_buffers[0]);
			m_allocator->DeviceFree(m_primitives.codes.d_buffers[1]);
			m_allocator->DeviceFree(m_primitives.indices.d_buffers[1]);
			m_allocator->DeviceFree(m_primitives.indices.d_buffers[0]);
			m_allocator->DeviceFree(m_primitives.boxes.d_buffers[1]);
			m_allocator->DeviceFree(d_temp_storage);
		}
		Bbox3f* getNodeBoxes()
		{
			return m_primitives.boxes.d_buffers[1];
		}
		Bbox3f* getLeafBoxes()
		{
			return m_primitives.boxes.d_buffers[0];
		}
		//get sorted indcies
		uint32* getIndices()
		{
			return m_primitives.indices.d_buffers[1];
		}
		template <typename Iterator, typename BboxIterator>
		void build(nih::Bbox3f globalBox, Iterator begin, Iterator end, BboxIterator boxBegin, BboxIterator boxEnd, DBVH* bvh);
	private:
		Primitives m_primitives;
		thrust::device_vector<Bvh_Node>* m_nodes;
		thrust::device_vector<Bvh_Node>* m_leaves;
		cub::CachingDeviceAllocator*  m_allocator; 
		void    *d_temp_storage;
		size_t  temp_storage_bytes;
	};
}
#include "cuda_klbvh_inline.h"