#include "MortonCode.h"
#include <iostream>
#include "math.h"
#include "GpuTimer.cuh"
#include "frustum.h"
using namespace nih; 
namespace cuda
{
	

	__device__ inline int Theta(uint32 i, uint32 j, uint32* keys, uint32 size)
	{
		if (j>size)
			return -1;
		uint32 a1 = keys[i];
		uint32 a2 = keys[j];
		if ( a1==a2)
		{ 
			return 32+__clz(i^j);
		}
		else
		{
			return __clz(a1^a2); 
		}
	}

	__global__ void buildKernel(Bvh_Node* nodes, Bvh_Node* leaves, uint32* keys, uint32* indices, uint32 n)
	{
		uint32 step = blockDim.x * gridDim.x;
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
			i < n; 
			i += step) 
		{

			int d = sgn(Theta(i,i+1,keys,n)-Theta(i,i-1,keys,n));
			//int d = signbit(1.0*Theta(i,i-1,keys,n)-Theta(i,i+1,keys,n));
			int tMin = Theta(i,i-d,keys,n);
			int lMax = 128;
			while(Theta(i,i+lMax*d,keys,n) > tMin)
				lMax *= 4;

			int l = 0;
			for(uint32 t = lMax/2; t>=1; t/=2)
			{
				if (Theta(i,i+(t+l)*d,keys,n) > tMin)
					l += t;
			}
			int j = i + l*d;
			int tNode = Theta(i,j,keys,n);
			int s = 0;

			for(uint32 t = ceil(l/2.0); t!=1; t=ceil(t/2.0))
			{
				if (Theta(i,i+(s+t)*d,keys,n) > tNode)
					s = s+t;
			}
			if (Theta(i,i+(s+1)*d,keys,n) > tNode)
				s = s+1;

			int gama  = i + s*d + nih::min(d,0);
			nodes[i].childIdx = gama;
			nodes[i].id = i;
			nodes[i].leafStart = nih::min(i,j);
			nodes[i].leafEnd = nih::max(i,j);
			nodes[i].isLeaf = false;
			if(nih::min(i,j) == gama) 
			{
				nodes[i].l_isleaf = true; 		
				leaves[gama].parentIdx = i;
				leaves[gama].isLeaf = true;
				leaves[gama].pid = indices[gama];
				leaves[gama].id = gama;
			}
			else
			{
				nodes[i].l_isleaf = false; 	
				nodes[gama].parentIdx = i;				
			}
			if (nih::max(i,j) == gama +1)
			{
				nodes[i].r_isleaf = true;	
				leaves[gama+1].parentIdx = i;
				leaves[gama+1].isLeaf = true;
				leaves[gama+1].pid = indices[gama+1];
				leaves[gama+1].id = gama+1;
			}
			else
			{
				nodes[i].r_isleaf = false;
				nodes[gama+1].parentIdx = i;				
			}
		}
	}

	__global__ void AssignAABBKernel(Bvh_Node* nodes, Bvh_Node* leaves, Bbox3f* nodeBoxes, Bbox3f* leafBoxes, uint32* counter, uint32 n)
	{
		uint32 step = blockDim.x * gridDim.x;
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
			i < n; 
			i += step) 
		{

			int j = leaves[i].parentIdx;
			while(j >=0)
			{
				if (atomicAdd(&(counter[j]),1) == 0)
				{					
					return;
				}
				else
				{
					Bbox3f lBox, rBox;
					Bvh_Node node = nodes[j];
					int lId = node.getChild(0);
					int rId = node.getChild(1);
					//pid 表示叶子代表图元在输入图元列表中的id
					if (node.l_isleaf) 
						lBox = leafBoxes[leaves[lId].pid];
					else
						lBox = nodeBoxes[lId];
					if (node.r_isleaf)
						rBox = leafBoxes[leaves[rId].pid];
					else
						rBox = nodeBoxes[rId];

					lBox.insert(rBox);
					nodeBoxes[j] = lBox;

					j = node.parentIdx;
					//atomicAdd(&(counter[j]),1);
				}
			}
		}
	}
	__global__ void AssignAABBKernel_shared(Bvh_Node* nodes, Bvh_Node* leaves, Bbox3f* nodeBoxes, Bbox3f* leafBoxes, uint32* indices, uint32* counter, uint32 n)
	{
		const int BLOCK_SIZE = 128;
		uint32 step = blockDim.x * gridDim.x;
		__shared__ uint32 s_counters[BLOCK_SIZE];


		for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
			i < n; 
			i += step) 
		{
			int I=threadIdx.x;
			s_counters[I] = counter[i];
			__syncthreads();

			int j = leaves[i].parentIdx;
			while(j >=0)
			{
				int current;
				if (j<BLOCK_SIZE-1)
					current = atomicAdd(&(s_counters[j]),1);
				else
					current = atomicAdd(&(counter[j]),1);
				if (current == 0)
				{					
					return;
				}
				else
				{
					Bbox3f lBox, rBox;
					Bvh_Node node = nodes[j];
					int lId = node.getChild(0);
					int rId = node.getChild(1);
					if (node.l_isleaf) 
						lBox = leafBoxes[indices[lId]];
					else
						lBox = nodeBoxes[lId];
					if (node.r_isleaf)
						rBox = leafBoxes[indices[rId]];
					else
						rBox = nodeBoxes[rId];

					lBox.insert(rBox);
					nodeBoxes[j] = lBox;

					j = node.parentIdx;
					//atomicAdd(&(counter[j]),1);
				}
			}
		}
	}



	KBvh_Builder::KBvh_Builder(thrust::device_vector<Bvh_Node>& nodes,
		thrust::device_vector<Bvh_Node>& leaves,
		cub::CachingDeviceAllocator& allocator):
	m_nodes(&nodes),
		m_leaves(&leaves),
		m_allocator(&allocator)
	{
		size_t n_points = nodes.size();

		CubDebugExit(m_allocator->DeviceAllocate((void**)&m_primitives.boxes.d_buffers[1], sizeof(Bbox3f) * n_points));

		CubDebugExit(m_allocator->DeviceAllocate((void**)&m_primitives.indices.d_buffers[1], sizeof(Vector3f) * n_points));
		CubDebugExit(m_allocator->DeviceAllocate((void**)&m_primitives.indices.d_buffers[0], sizeof(Vector3f) * n_points));

		CubDebugExit(m_allocator->DeviceAllocate((void**)&m_primitives.codes.d_buffers[0], sizeof(uint32) * n_points));
		CubDebugExit(m_allocator->DeviceAllocate((void**)&m_primitives.codes.d_buffers[1], sizeof(uint32) * n_points));

		/*Allocate temporary storage*/
		temp_storage_bytes  = 0;
		d_temp_storage     = NULL;
		CubDebugExit(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, m_primitives.codes, m_primitives.indices, n_points));
		//CubDebugExit(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, m_primitives.codes,  n_points));
		CubDebugExit(m_allocator->DeviceAllocate(&d_temp_storage, temp_storage_bytes));


	}
	template <typename Iterator, typename BboxIterator>
	void KBvh_Builder::build(Bbox3f globalBox, Iterator begin, Iterator end, BboxIterator boxBegin, BboxIterator boxEnd, DBVH* d_bvh)
	{

		size_t n_points = end - begin;
		m_primitives.boxes.d_buffers[0] = thrust::raw_pointer_cast(&(*boxBegin));

		//std::cout<<"assign morton code"<<std::endl;
		//compute the Morton code for each point
		GpuTimer timer;
		//timer.Start();
		thrust::transform(
			begin,
			begin + n_points,
			thrust::device_ptr<uint32>(m_primitives.codes.d_buffers[0]),
			morton_functor<uint32>( globalBox ) );

		//timer.Stop();
		//std::cout<<"assign morton code "<<timer.ElapsedMillis()<<std::endl;

		/*uint32 test_codes[] = {1,2,4,5,19,24,25,30};
		cudaMemcpy(m_primitives.codes.d_buffers[0],test_codes,sizeof(uint32)*sizeof(test_codes),cudaMemcpyHostToDevice);*/
		//sort by motron code
		//timer.Start();
		thrust::copy(
			thrust::counting_iterator<uint32>(0),
			thrust::counting_iterator<uint32>(0) + n_points,
			thrust::device_ptr<uint32>(m_primitives.indices.d_buffers[0]) );
		CubDebugExit(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, m_primitives.codes, m_primitives.indices, n_points));
		//CubDebugExit(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, m_primitives.codes,n_points));
		//timer.Stop();
		//std::cout<<"sort code "<<timer.ElapsedMillis()<<std::endl;
		/*uint32* h_codes = (uint32*)malloc(n_points*sizeof(uint32));
		cudaMemcpy(h_codes,m_primitives.codes.d_buffers[1],n_points*sizeof(uint32),cudaMemcpyDeviceToHost);
		for(int i=0; i< n_points; i++)
		{
		std::cout<<h_codes[i]<<std::endl;
		}
		free(h_codes);*/

		const uint32 BLOCK_SIZE = 128;
		int numSMs;
		cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
		size_t max_blocks = 65535;
		const size_t n_blocks   = nih::min( max_blocks, (n_points + (BLOCK_SIZE*numSMs)-1) / (BLOCK_SIZE*numSMs) );
		Bvh_Node* nodes = thrust::raw_pointer_cast(&(*m_nodes).front());
		Bvh_Node* leaves = thrust::raw_pointer_cast(&(*m_leaves).front());
		//timer.Start();
		buildKernel<<<n_blocks*numSMs,BLOCK_SIZE>>>(nodes,leaves,m_primitives.codes.d_buffers[1],m_primitives.indices.d_buffers[1],n_points-1);
		//timer.Stop();
		//std::cout<<"build tree "<<timer.ElapsedMillis()<<std::endl;
		thrust::device_vector<uint32> d_counters(n_points,0);
		uint32* rawPtrCounters = thrust::raw_pointer_cast(&d_counters.front());
		//timer.Start();
		AssignAABBKernel<<<n_blocks*numSMs,BLOCK_SIZE>>>(nodes,leaves,m_primitives.boxes.d_buffers[1],m_primitives.boxes.d_buffers[0],rawPtrCounters, n_points);
		//timer.Stop();
		//std::cout<<"assign aabb "<<timer.ElapsedMillis()<<std::endl;

		d_bvh->nodes = nodes;
		d_bvh->leaves = leaves;
		d_bvh->nodeBoxes = m_primitives.boxes.d_buffers[1];
		d_bvh->leafBoxes = m_primitives.boxes.d_buffers[0];
	}
}