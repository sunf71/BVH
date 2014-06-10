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

/*! \file reduce.h
 *   \brief Defines utility function to reduce a set of values attached
 *          to the elements in the leaves of a tree.
 */

#pragma once

#include "basic/types.h"
#include "tree/model.h"
#include <iterator>

namespace nih {
namespace cuda {

/*! \addtogroup trees Trees
 *  \{
 */

///
/// Reduce a bunch of values attached to the elemens in the leaves of a tree.
/// The Tree template type has to provide the following breadth-first tree
/// interface:
///
/// \code
///
/// struct Tree
/// {
///     // return the number of levels
///     uint32 get_level_count() const;
///
///     // return the index of the first node in the i-th level
///     uint32 get_level(const uint32 i) const;
///
///     // retrieve a node
///     node_type get_node(const uint32 index) const;
///
///     // return the number of leaves
///     uint32 get_leaf_count() const;
///
///     // retrieve a leaf
///     uint2 get_leaf(const uint32 index) const;
/// };
///
/// \endcode
///
/// The following code snippet illustrates an example usage:
///
/// \code
///
/// #include <nih/tree/cuda/tree_reduce.h>
/// #include <nih/tree/model.h>
///
/// struct merge_op
/// {
///     NIH_HOST_DEVICE Bbox4f operator() (
///         const Bbox4f op1,
///         const Bbox4f op2) const { return Bbox4f( op1, op2 ); }
/// };
///
/// // compute the bboxes of a tree
/// void compute_bboxes(
///     uint32      node_count,     // input tree nodes
///     uint32      leaf_count,     // input tree leaves
///     uint32      level_count,    // input tree levels
///     Bvh_node*   nodes,          // input tree nodes, device pointer
///     uint2*      leaves,         // input tree leaves, device pointer
///     uint32*     levels,         // input tree levels, host pointer
///     Bbox4f*     bboxes,         // input primitive bboxes, device pointer
///     Bbox4f*     node_bboxes,    // output node bboxes, device pointer
///     Bbox4f*     leaf_bboxes)    // output leaf bboxes, device pointer
/// {
///     // instantiate a breadth-first tree view
///     BFTree<Bvh_node*,device_domain> bvh(
///         nodes,
///         leaf_count,
///         leaves,
///         level_count,
///         levels );
///
///     // compute a tree reduction
///     cuda::tree_reduce(
///         bvh,
///         bboxes,
///         leaf_bboxes,
///         node_bboxes,
///         merge_op(),
///         Bbox4f() );
/// }
///
/// \endcode
///

template <typename Tree, typename Input_iterator, typename Output_iterator, typename Operator>
void tree_reduce(
    const Tree              tree,
    const Input_iterator    in_values,
    Output_iterator         leaf_values,
    Output_iterator         node_values,
    const Operator          op,
    const typename std::iterator_traits<Output_iterator>::value_type def_value);

///
/// Reduce a bunch of values attached to the leaves of a tree.
///
template <typename Tree, typename Input_iterator, typename Output_iterator, typename Operator>
void tree_reduce(
    const Tree              tree,
    const Input_iterator    in_values,
    Output_iterator         node_values,
    const Operator          op);

/*! \}
 */

} // namespace cuda
} // namespace nih

#include "tree/cuda/reduce_inline.h"

