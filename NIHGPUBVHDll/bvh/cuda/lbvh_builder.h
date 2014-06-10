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

/*! \file lbvh_builder.h
 *   \brief Interface for a CUDA-based LBVH builder.
 */

#pragma once

#include "../../bvh/bvh.h"
#include "../../linalg/vector.h"
#include "../../linalg/bbox.h"
#include "../../bintree/bintree_node.h"
#include "../../bintree/cuda/bintree_gen_context.h"
#include <thrust/device_vector.h>

namespace nih {
namespace cuda {

/*! \addtogroup bvh Bounding Volume Hierarchies
 *  \{
 */

///
/// GPU-based Linar BVH builder
///
/// This class provides the context to generate LBVHs on the GPU
/// starting from a set of unordered points.
/// The output is a set of nodes with the corresponding leaves and
/// a set of primitive indices into the input set of points.
/// The output leaves will specify contiguous ranges into this index.
///
/// \tparam Integer     an integer type that determines the number
///                     of bits used to compute the points' Morton codes.
///                     Accepted values are uint32 and uint64.
///
/// The following code snippet shows how to use this builder:
///
/// \code
///
/// #include <nih/bvh/cuda/lbvh_builder.h>
///
/// thrust::device_vector<Vector3f> points;
/// ... // code to fill the input vector of points
///
/// thrust::device_vector<Bvh_node> bvh_nodes;
/// thrust::device_vector<uint2>    bvh_leaves;
/// thrust::device_vector<uint32>   bvh_index;
///
/// nih::cuda::LBVH_builder<uint64> builder( bvh_nodes, bvh_leaves, bvh_index );
/// builder.build(
///     Bbox3f( Vector3f(0.0f), Vector3f(1.0f) ),   // suppose all bboxes are in [0,1]^3
///     points.begin(),                             // begin iterator
///     points.end(),                               // end iterator
///     4 );                                        // target 4 objects per leaf
/// 
///  \endcode
///
template <typename Integer>
struct LBVH_builder
{
    /// constructor
    ///
    /// \param nodes        output nodes array
    /// \param leaves       output leaf array
    /// \param index        output index array
    LBVH_builder(
        thrust::device_vector<Bvh_node>&         nodes,
        thrust::device_vector<uint2>&            leaves,
        thrust::device_vector<uint32>&           index) :
        m_nodes( &nodes ), m_leaves( &leaves ), m_index( &index ) {}

    /// build a bvh given a set of points that will be reordered in-place
    ///
    /// \param bbox             global bbox
    /// \param points_begin     beginning of the point sequence to sort
    /// \param points_end       end of the point sequence to sort
    /// \param max_leaf_size    maximum leaf size
    template <typename Iterator>
    void build(
        const Bbox3f    bbox,
        const Iterator  points_begin,
        const Iterator  points_end,
        const uint32    max_leaf_size);

	/// build a bvh given a set of  bboxes with center points that will be reordered in-place
    ///
    /// \param bbox             global bbox
    /// \param points_begin     beginning of the point sequence to sort
    /// \param points_end       end of the point sequence to sort
	/// \param Bbox_begin	beginning of the Bbox sequence to sort
	/// \param Bbox_end	end of the Bbox sequence to sort
    /// \param max_leaf_size    maximum leaf size
    template <typename Iterator, typename BboxIterator>
    void build(
        const Bbox3f    bbox,
        const Iterator  points_begin,
        const Iterator  points_end,
		const BboxIterator Bbox_begin,
		const BboxIterator Bbox_end,
        const uint32    max_leaf_size);

    thrust::device_vector<Bvh_node>*    m_nodes;
    thrust::device_vector<uint2>*       m_leaves;
    thrust::device_vector<uint32>*      m_index;
    thrust::device_vector<Integer>      m_codes;
    uint32                              m_levels[64];
    Bbox3f                              m_bbox;
    uint32                              m_node_count;
    uint32                              m_leaf_count;

    cuda::Bintree_gen_context           m_kd_context;
};


struct bbox_functor
{
    NIH_HOST_DEVICE Bbox4f operator() (
        const Vector4f op1,
        const Vector4f op2) const
    {
        Bbox4f result;
        result.insert( op1 );
        result.insert( op2 );
        return result;
    }
    NIH_HOST_DEVICE Bbox4f operator() (
        const Bbox4f op1,
        const Bbox4f op2) const
    {
        Bbox4f result;
        result.insert( op1 );
        result.insert( op2 );
        return result;
    }
};
/*! \}
 */

} // namespace cuda
} // namespace nih

#include "../../bvh/cuda/lbvh_builder_inline.h"