/*
 * Copyright (c) 2010-2011, NVIDIA Corporation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of NVIDIA Corporation nor the
 *     names of its contributors may be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "../../basic/utils.h"
#include "../../bvh/cuda/lbvh_context.h"
#include "../../bintree/cuda/bintree_gen.h"
#include "../../bits/morton.h"
#include <thrust/sort.h>



template <typename keyType, typename valType>
void CUBSort(thrust::device_ptr<keyType> keyStart, 
	                   thrust::device_ptr<keyType> keyEnd,
					   thrust::device_ptr<valType> valStart)
{
	using namespace cub;
	CachingDeviceAllocator  allocator(true); 
	unsigned int num_items = keyEnd - keyStart +1;
	keyType* rawPtr_dKeyVec = thrust::raw_pointer_cast(keyStart);
	valType* rawPtr_dValVec = thrust::raw_pointer_cast(valStart);
	 // Allocate device arrays
    DoubleBuffer<keyType> d_keys;
    DoubleBuffer<valType>   d_values;
	d_keys.d_buffers[0] = rawPtr_dKeyVec;
    //CubDebugExit(allocator.DeviceAllocate((void**)&d_keys.d_buffers[0], sizeof(keyType) * num_items));
    CubDebugExit(allocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(keyType) * num_items));
    //CubDebugExit(allocator.DeviceAllocate((void**)&d_values.d_buffers[0], sizeof(valType) * num_items));
    CubDebugExit(allocator.DeviceAllocate((void**)&d_values.d_buffers[1], sizeof(valType) * num_items));
	d_values.d_buffers[0] = rawPtr_dValVec;
	// Allocate temporary storage
    size_t  temp_storage_bytes  = 0;
    void    *d_temp_storage     = NULL;
	
    CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items));
    CubDebugExit(allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

	

	
	//cudaMemcpy(d_keys.d_buffers[d_keys.selector],rawPtr_dKeyVec, sizeof(keyType) * num_items, cudaMemcpyDeviceToDevice);
	//cudaMemcpy(d_values.d_buffers[d_values.selector],rawPtr_dValVec, sizeof(valType) * num_items, cudaMemcpyDeviceToDevice);
	//cudaEvent_t start, stop; 
	//cudaEventCreate( &start );
	//cudaEventCreate( &stop );
	//float time = 0.0f;
	//cudaEventRecord( start, 0 );
	// Run
    CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items));

	//cudaEventRecord( stop, 0 );
	//cudaEventSynchronize( stop );
	//cudaEventElapsedTime( &time, start, stop );

	//printf("sort time elapsed: %fms\n", time);
	cudaMemcpy(rawPtr_dKeyVec, d_keys.Current(), sizeof(keyType) * num_items, cudaMemcpyDeviceToDevice);
	cudaMemcpy(rawPtr_dValVec, d_values.Current(), sizeof(valType) * num_items, cudaMemcpyDeviceToDevice);

	//clean
    //if (d_keys.d_buffers[0]) CubDebugExit(allocator.DeviceFree(d_keys.d_buffers[0]));
   if (d_keys.d_buffers[1]) CubDebugExit(allocator.DeviceFree(d_keys.d_buffers[1]));
    //if (d_values.d_buffers[0]) CubDebugExit(allocator.DeviceFree(d_values.d_buffers[0]));
   if (d_values.d_buffers[1]) CubDebugExit(allocator.DeviceFree(d_values.d_buffers[1]));
   if (d_temp_storage) CubDebugExit(allocator.DeviceFree(d_temp_storage));
}

namespace nih {
namespace cuda {

namespace lbvh {

    template <typename Integer>
    struct Morton_bits {};

    template <>
    struct Morton_bits<uint32> { static const uint32 value = 30u; };

    template <>
    struct Morton_bits<uint64> { static const uint32 value = 60u; };

};

// build a Linear BVH given a set of points
template <typename Integer>
template <typename Iterator>
void LBVH_builder<Integer>::build(
    const Bbox3f    bbox,
    const Iterator  points_begin,
    const Iterator  points_end,
    const uint32    max_leaf_size)
{ 
    typedef cuda::Bintree_gen_context::Split_task Split_task;

    const uint32 n_points = uint32( points_end - points_begin );

    m_bbox = bbox;
    need_space( m_codes, n_points );
    need_space( *m_index, n_points );
    need_space( *m_leaves, n_points );

    // compute the Morton code for each point
    thrust::transform(
        points_begin,
        points_begin + n_points,
        m_codes.begin(),
        morton_functor<Integer>( bbox ) );

    // setup the point indices, from 0 to n_points-1
    thrust::copy(
        thrust::counting_iterator<uint32>(0),
        thrust::counting_iterator<uint32>(0) + n_points,
        m_index->begin() );

    // sort the indices by Morton code
    // TODO: use Duane's library directly here... this is doing shameful allocations!
    thrust::sort_by_key(
        m_codes.begin(),
        m_codes.begin() + n_points,
        m_index->begin() );

    // generate a kd-tree
    LBVH_context tree( m_nodes, m_leaves );

    const uint32 bits = lbvh::Morton_bits<Integer>::value;

    generate(
        m_kd_context,
        n_points,
        thrust::raw_pointer_cast( &m_codes.front() ),
        bits,
        max_leaf_size,
        false,
        tree );

    m_leaf_count = m_kd_context.m_leaves;
    m_node_count = m_kd_context.m_nodes;

    for (uint32 level = 0; level <= bits; ++level)
        m_levels[ bits - level ] = m_kd_context.m_levels[ level ];
}

// build a Linear BVH given a set of Bboxes with center points
template <typename Integer>
template <typename Iterator, typename BboxIterator>
void LBVH_builder<Integer>::build(
        const Bbox3f    bbox,
        const Iterator  points_begin,
        const Iterator  points_end,
		const BboxIterator Bbox_begin,
		const BboxIterator Bbox_end,
        const uint32    max_leaf_size)
{
	 typedef cuda::Bintree_gen_context::Split_task Split_task;

    const uint32 n_points = uint32( points_end - points_begin );

    m_bbox = bbox;
    need_space( m_codes, n_points );
    need_space( *m_index, n_points );
    need_space( *m_leaves, n_points );

    // compute the Morton code for each point
	nih::GpuTimer timer;
	//timer.Start();
    thrust::transform(
        points_begin,
        points_begin + n_points,
        m_codes.begin(),
        morton_functor<Integer>( bbox ) );
	/*timer.Stop();
	printf("assign morton code %f\n",timer.ElapsedMillis());*/
	//thrust::device_vector<Integer> unsorted_codes(m_codes);
    // setup the point indices, from 0 to n_points-1
   /* thrust::copy(
        thrust::counting_iterator<uint32>(0),
        thrust::counting_iterator<uint32>(0) + n_points,
        m_index->begin() );*/

    // sort the indices by Morton code
    // TODO: use Duane's library directly here... this is doing shameful allocations!
	
	timer.Start();
   /* thrust::sort_by_key(
        m_codes.begin(),
        m_codes.begin() + n_points,
        m_index->begin() );*/
	//CUBSort(&m_codes.front(),&m_codes.back(),&m_index->front());

	//
	//thrust::device_ptr<Bbox4f> dev_ptr = &(*Bbox_begin);
	//CUBSort(&m_codes.front(),&m_codes.back(), dev_ptr);
	thrust::sort_by_key(
		m_codes.begin(),
		m_codes.begin() + n_points,
		Bbox_begin);
	timer.Stop();
	printf("radix sort %f\n",timer.ElapsedMillis());

    // generate a kd-tree
    LBVH_context tree( m_nodes, m_leaves );

    const uint32 bits = lbvh::Morton_bits<Integer>::value;

    generate(
        m_kd_context,
        n_points,
        thrust::raw_pointer_cast( &m_codes.front() ),
        bits,
        max_leaf_size,
        false,
        tree );

    m_leaf_count = m_kd_context.m_leaves;
    m_node_count = m_kd_context.m_nodes;

    for (uint32 level = 0; level <= bits; ++level)
        m_levels[ bits - level ] = m_kd_context.m_levels[ level ];
}

template <typename Integer>
template <typename Iterator, typename BboxIterator>
void LBVH_builder<Integer>::build(
        const Bbox3f    bbox,
        const Iterator  points_begin,
        const Iterator  points_end,
		const BboxIterator Bbox_begin,
		const BboxIterator Bbox_end,
        const uint32    max_leaf_size,
		cub::DoubleBuffer<uint32> & d_keys,
		cub::DoubleBuffer<Bbox4f> & d_values,
		void    **d_temp_storage,
		size_t temp_storage_bytes)
{
	using namespace cub;
	 typedef cuda::Bintree_gen_context::Split_task Split_task;

    const uint32 n_points = uint32( points_end - points_begin );

    m_bbox = bbox;
    need_space( m_codes, n_points );
    need_space( *m_index, n_points );
    need_space( *m_leaves, n_points );

    // compute the Morton code for each point
	/*nih::GpuTimer timer;*/
	//timer.Start();
    thrust::transform(
        points_begin,
        points_begin + n_points,
        m_codes.begin(),
        morton_functor<Integer>( bbox ) );
	/*timer.Stop();
	printf("assign morton code %f\n",timer.ElapsedMillis());*/
	//thrust::device_vector<Integer> unsorted_codes(m_codes);
    // setup the point indices, from 0 to n_points-1
   /* thrust::copy(
        thrust::counting_iterator<uint32>(0),
        thrust::counting_iterator<uint32>(0) + n_points,
        m_index->begin() );*/

    // sort the indices by Morton code
    // TODO: use Duane's library directly here... this is doing shameful allocations!
	
	/*timer.Start();*/
   /* thrust::sort_by_key(
        m_codes.begin(),
        m_codes.begin() + n_points,
        m_index->begin() );*/
	//CUBSort(&m_codes.front(),&m_codes.back(),&m_index->front());

	//
	d_keys.d_buffers[0] = thrust::raw_pointer_cast(&m_codes.front());
	//thrust::device_ptr<Bbox4f> dev_ptr = &(*Bbox_begin);
	//CUBSort(&m_codes.front(),&m_codes.back(), dev_ptr);
	/*thrust::sort_by_key(
		m_codes.begin(),
		m_codes.begin() + n_points,
		Bbox_begin);*/
	CubDebugExit(DeviceRadixSort::SortPairs(*d_temp_storage, temp_storage_bytes, d_keys, d_values, n_points));
	/*timer.Stop();
	printf("radix sort %f\n",timer.ElapsedMillis());*/

    // generate a kd-tree
    LBVH_context tree( m_nodes, m_leaves );

    const uint32 bits = lbvh::Morton_bits<Integer>::value;

    generate(
        m_kd_context,
        n_points,
        thrust::raw_pointer_cast( d_keys.d_buffers[1] ),
        bits,
        max_leaf_size,
        false,
        tree );

    m_leaf_count = m_kd_context.m_leaves;
    m_node_count = m_kd_context.m_nodes;

    for (uint32 level = 0; level <= bits; ++level)
        m_levels[ bits - level ] = m_kd_context.m_levels[ level ];
}
} 

// namespace cuda
} // namespace nih
