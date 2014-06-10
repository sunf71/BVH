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

#pragma once

#include "basic/types.h"


namespace nih {
namespace cuda {

namespace treereduce {

// reduce leaf values
template <uint32 BLOCK_SIZE, typename Tree, typename Input_iterator, typename Output_iterator, typename Operator>
__global__ void reduce_leaves_kernel(
    const Tree              tree,
    const uint32            n_leaves,
    const Input_iterator    in_values,
    Output_iterator         out_values,
    const Operator          op,
    const typename std::iterator_traits<Output_iterator>::value_type def_value)
{
    const uint32 grid_size = gridDim.x * blockDim.x;

    // loop through all logical blocks associated to this physical one
    for (uint32 base_idx = blockIdx.x * BLOCK_SIZE;
        base_idx < n_leaves;
        base_idx += grid_size)
    {
        const uint32 leaf_id = threadIdx.x + base_idx;

        if (leaf_id < n_leaves)
        {
            const uint2 leaf = tree.get_leaf( leaf_id );
            const uint32 begin = leaf.x;
            const uint32 end   = leaf.y;

            typename std::iterator_traits<Output_iterator>::value_type value = def_value;
            for (uint32 i = begin; i < end; ++i)
               // value = op( value, thrust::detail::backend::dereference( in_values + i ) );
			 value = op( value, in_values[i] );

            //thrust::detail::backend::dereference( out_values, leaf_id ) = value;
			out_values[leaf_id] = value;
        }
    }
}

// reduce leaf values
template <typename Tree, typename Input_iterator, typename Output_iterator, typename Operator>
void reduce_leaves(
    const Tree              tree,
    const Input_iterator    in_values,
    Output_iterator         out_values,
    const Operator          op,
    const typename std::iterator_traits<Output_iterator>::value_type def_value)
{
    const uint32 n_leaves = tree.get_leaf_count();

    const uint32 BLOCK_SIZE = 128;
    //const size_t max_blocks = thrust::detail::backend::cuda::arch::max_active_blocks(reduce_leaves_kernel<BLOCK_SIZE, Tree, Input_iterator, Output_iterator, Operator>, BLOCK_SIZE, 0);
    const size_t max_blocks = 128;
	const size_t n_blocks   = nih::min( max_blocks, (n_leaves + BLOCK_SIZE-1) / BLOCK_SIZE );

    reduce_leaves_kernel<BLOCK_SIZE> <<<n_blocks,BLOCK_SIZE>>> (
        tree,
        n_leaves,
        in_values,
        out_values,
        op,
        def_value );

    cudaThreadSynchronize();
}

// reduce a level
template <uint32 BLOCK_SIZE, typename Tree, typename Leaf_iterator, typename Output_iterator, typename Operator>
__global__ void reduce_level_kernel(
    const Tree              tree,
    const uint32            begin,
    const uint32            end,
    Leaf_iterator           leaf_values,
    Output_iterator         out_values,
    const Operator          op)
{
    const uint32 grid_size = gridDim.x * blockDim.x;

    // loop through all logical blocks associated to this physical one
    for (uint32 base_idx = blockIdx.x * BLOCK_SIZE + begin;
        base_idx < end;
        base_idx += grid_size)
    {
        const uint32 node_id = threadIdx.x + base_idx;

        if (node_id < end)
        {
            const typename Tree::node_type node = tree.get_node( node_id );

            if (node.is_leaf())
            {
                // copy the corresponding leaf value
                const uint32 leaf_index = node.get_leaf_index();
                thrust::detail::backend::dereference( out_values + node_id ) = thrust::detail::backend::dereference( leaf_values + leaf_index );
            }
            else
            {
                // reduce all child values
                const uint32 n_children = node.get_child_count();

                typename std::iterator_traits<Output_iterator>::value_type value =
                    thrust::detail::backend::dereference( out_values, node.get_child(0) );
                for (uint32 i = 1; i < n_children; ++i)
                    value = op( value, thrust::detail::backend::dereference( out_values, node.get_child(i) ) );

                thrust::detail::backend::dereference( out_values, node_id ) = value;
            }
        }
    }
}

// reduce leaf values
template <typename Tree, typename Leaf_iterator, typename Output_iterator, typename Operator>
void reduce_level(
    const Tree              tree,
    const uint32            begin,
    const uint32            end,
    Leaf_iterator           leaf_values,
    Output_iterator         out_values,
    const Operator          op)
{
    const uint32 n_entries = end - begin;

    const uint32 BLOCK_SIZE = 128;
    const size_t max_blocks = thrust::detail::backend::cuda::arch::max_active_blocks(reduce_level_kernel<BLOCK_SIZE, Tree, Leaf_iterator, Output_iterator, Operator>, BLOCK_SIZE, 0);
    const size_t n_blocks   = nih::min( max_blocks, (n_entries + BLOCK_SIZE-1) / BLOCK_SIZE );

    reduce_level_kernel<BLOCK_SIZE> <<<n_blocks,BLOCK_SIZE>>> (
        tree,
        begin,
        end,
        leaf_values,
        out_values,
        op );

    cudaThreadSynchronize();
}

template <typename Tree_type>
struct reduce {};

template <>
struct reduce<breadth_first_tree>
{
    //
    // Reduce a bunch of values attached to the leaves of a breadth-first tree.
    // The tree is supposed to be laid out in a breadth-first fashion,
    // with the beginning of the nodes for level m specified by the map
    // levels[m] - where 0 is the root level.
    //
    template <typename Tree, typename Input_iterator, typename Output_iterator, typename Operator>
    static void dispatch(
        const Tree              tree,
        const Input_iterator    in_values,
        Output_iterator         leaf_values,
        Output_iterator         node_values,
        const Operator          op,
        const typename std::iterator_traits<Output_iterator>::value_type def_value)
    {
        reduce_leaves(
            tree,
            in_values,
            leaf_values,
            op,
            def_value );

        const uint32 n_levels = tree.get_level_count();

        uint32 offset = 0;

        for (int32 level = n_levels-1; level >= 0; --level)
        {
            const uint32 level_begin = tree.get_level(level);
            const uint32 level_end   = tree.get_level(level+1);
            const uint32 level_size  = level_end - level_begin;

            if (level_size == 0)
                continue;

            reduce_level(
                tree,
                level_begin,
                level_end,
                leaf_values,
                node_values,
                op );

            offset += level_size;
        }
    }

    //
    // Reduce a bunch of values attached to the leaves of a breadth-first tree.
    // The tree is supposed to be laid out in a breadth-first fashion,
    // with the beginning of the nodes for level m specified by the map
    // levels[m] - where 0 is the root level.
    //
    template <typename Tree, typename Input_iterator, typename Output_iterator, typename Operator>
    static void dispatch(
        const Tree              tree,
        const Input_iterator    leaf_values,
        Output_iterator         node_values,
        const Operator          op)
    {
        const uint32 n_levels = tree.get_level_count();

        uint32 offset = 0;

        for (int32 level = n_levels-1; level >= 0; --level)
        {
            const uint32 level_begin = tree.get_level(level);
            const uint32 level_end   = tree.get_level(level+1);
            const uint32 level_size  = level_end - level_begin;

            if (level_size == 0)
                continue;

            reduce_level(
                tree,
                level_begin,
                level_end,
                leaf_values,
                node_values,
                op );

            offset += level_size;
        }
    }
};

template <>
struct reduce<depth_first_tree>
{
    // TODO: implement this!
};

} // namespace tree_reduce

//
// Reduce a bunch of values attached to the elemens in the leaves of a tree.
//
template <typename Tree, typename Input_iterator, typename Output_iterator, typename Operator>
void tree_reduce(
    const Tree              tree,
    const Input_iterator    in_values,
    Output_iterator         leaf_values,
    Output_iterator         node_values,
    const Operator          op,
    const typename std::iterator_traits<Output_iterator>::value_type def_value)
{
    treereduce::reduce<typename Tree::tree_type>::dispatch(
        tree,
        in_values,
        leaf_values,
        node_values,
        op,
        def_value );
}

//
// Reduce a bunch of values attached to the leaves of a tree.
//
template <typename Tree, typename Input_iterator, typename Output_iterator, typename Operator>
void tree_reduce(
    const Tree              tree,
    const Input_iterator    in_values,
    Output_iterator         node_values,
    const Operator          op)
{
    treereduce::reduce<typename Tree::tree_type>::dispatch(
        tree,
        in_values,
        node_values,
        op );
}

} // namespace cuda
} // namespace nih
