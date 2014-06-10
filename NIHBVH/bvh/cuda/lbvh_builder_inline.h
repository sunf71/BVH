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

#include "bvh/cuda/lbvh_context.h"
#include "bintree/cuda/bintree_gen.h"
#include "bits/morton.h"
#include <thrust/sort.h>

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

} // namespace cuda
} // namespace nih
