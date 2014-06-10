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

#include "../../basic/functors.h"
#include "../../basic/algorithms.h"
#include "../../basic/cuda/scan.h"
#include "../../basic/utils.h"

namespace nih {
namespace cuda {
namespace bintree {

typedef Bintree_gen_context::Split_task Split_task;

// find the most significant bit smaller than start by which code0 and code1 differ
template <typename Integer>
FORCE_INLINE NIH_HOST_DEVICE int32 find_leading_bit_difference(
    const  int32  start_level,
    const Integer code0,
    const Integer code1)
{
	
    int32 level = start_level;

    while (level >= 0)
    {
        const Integer mask = Integer(1u) << level;

        if ((code0 & mask) !=
            (code1 & mask))
            break;

        --level;
    }
    return level;
}

// do a single kd-split for all nodes in the input task queue, and generate
// a corresponding list of output tasks
template <uint32 BLOCK_SIZE, typename Tree, typename Integer>
__global__ void split_kernel(
    Tree                tree,
    const uint32        max_leaf_size,
    const bool          keep_singletons,
    const uint32        grid_size,
    const Integer*      codes,
    const uint32        in_tasks_count,
    const Split_task*   in_tasks,
    const uint32*       in_skip_nodes,
    uint32*             out_tasks_count,
    Split_task*         out_tasks,
    uint32*             out_skip_nodes,
    const uint32        out_nodes_count,
    uint32*             out_leaf_count)
{
    const uint32 LOG_WARP_SIZE = 5;
    const uint32 WARP_SIZE = 1u << LOG_WARP_SIZE;

    volatile __shared__ uint32 warp_offset[ BLOCK_SIZE >> LOG_WARP_SIZE ];

    const uint32 warp_tid = threadIdx.x & (WARP_SIZE-1);
    const uint32 warp_id  = threadIdx.x >> LOG_WARP_SIZE;

    volatile __shared__ uint32 sm_red[ BLOCK_SIZE * 2 ];
    volatile uint32* warp_red = sm_red + WARP_SIZE * 2 * warp_id;

    // loop through all logical blocks associated to this physical one
    for (uint32 base_idx = blockIdx.x * BLOCK_SIZE;
                base_idx < in_tasks_count;
                base_idx += grid_size)
    {
        uint32 output_count = 0;
        uint32 split_index;

        const uint32 task_id = threadIdx.x + base_idx;

        uint32 node;
        uint32 begin;
        uint32 end;
        uint32 level;
        uint32 skip_node;

        // check if the task id is in range, and if so try to find its split plane
        if (task_id < in_tasks_count)
        {
            const Split_task in_task = in_tasks[ task_id ];

            node  = in_task.m_node;
            begin = in_task.m_begin;
            end   = in_task.m_end;
            level = in_task.m_input;

            skip_node = in_skip_nodes[ task_id ];

            if (!keep_singletons)
            {
                level = find_leading_bit_difference(
                    level,
                    codes[begin],
                    codes[end-1] );
            }

            // check whether the input node really needs to be split
            if (end - begin > max_leaf_size && level != uint32(-1))
            {
                // find the "partitioning pivot" using a binary search
                split_index = find_pivot(
                    codes + begin,
                    end - begin,
                    mask_and<Integer>( Integer(1u) << level ) ) - codes;

                output_count = (split_index == begin || split_index == end) ? 1u : 2u;
            }
        }

        const uint32 task_offset = cuda::alloc( output_count, out_tasks_count, warp_tid, warp_red, warp_offset + warp_id );
        const uint32 node_offset = out_nodes_count + task_offset;
        const uint32 first_end   = (output_count == 1) ? end       : split_index;
        const uint32 first_skip  = (output_count == 1) ? skip_node : node_offset+1;

        if (output_count >= 1) { out_tasks[ task_offset+0 ] = Split_task( node_offset+0, begin, first_end, level-1 ); out_skip_nodes[ task_offset+0 ] = first_skip; }
        if (output_count == 2) { out_tasks[ task_offset+1 ] = Split_task( node_offset+1, split_index, end, level-1 ); out_skip_nodes[ task_offset+1 ] = skip_node; }

        const bool generate_leaf = (output_count == 0 && task_id < in_tasks_count);

        // count how many leaves we need to generate
        const uint32 leaf_index = cuda::alloc<1>( generate_leaf, out_leaf_count, warp_tid, warp_offset + warp_id );

        // write the parent node
        if (task_id < in_tasks_count)
        {
            tree.write_node(
                node,
                output_count ? split_index != begin : false,
                output_count ? split_index != end   : false,
                output_count ? node_offset          : leaf_index,
                skip_node,
                level,
                begin,
                end,
                output_count ? split_index : uint32(-1) );

            // make a leaf if necessary
            if (output_count == 0)
                tree.write_leaf( leaf_index, begin, end );
        }
    }
}
// generate a leaf for each task
template <typename Tree, uint32 BLOCK_SIZE>
__global__ void gen_leaves_kernel(
    Tree                tree,
    const uint32        grid_size,
    const uint32        in_tasks_count,
    const Split_task*   in_tasks,
    const uint32*       in_skip_nodes,
    uint32*             out_leaf_count)
{
    const uint32 LOG_WARP_SIZE = 5;
    const uint32 WARP_SIZE = 1u << LOG_WARP_SIZE;

    __shared__ uint32 warp_offset[ BLOCK_SIZE >> LOG_WARP_SIZE ];

    const uint32 warp_tid = threadIdx.x & (WARP_SIZE-1);
    const uint32 warp_id  = threadIdx.x >> LOG_WARP_SIZE;

    // loop through all logical blocks associated to this physical one
    for (uint32 base_idx = blockIdx.x * BLOCK_SIZE;
                base_idx < in_tasks_count;
                base_idx += grid_size)
    {
        const uint32 task_id = threadIdx.x + base_idx;

        uint32 node;
        uint32 begin;
        uint32 end;
        uint32 level;
        uint32 skip_node;

        // check if the task id is in range, and if so try to find its split plane
        if (task_id < in_tasks_count)
        {
            const Split_task in_task = in_tasks[ task_id ];

            node  = in_task.m_node;
            begin = in_task.m_begin;
            end   = in_task.m_end;
            level = in_task.m_input;
            skip_node = in_skip_nodes[ task_id ];
        }

        // alloc output slots
        uint32 leaf_index = cuda::alloc<1>( task_id < in_tasks_count, out_leaf_count, warp_tid, warp_offset + warp_id );

        // write the parent node
        if (task_id < in_tasks_count)
        {
            tree.write_node( node, false, false, leaf_index, skip_node, level, begin, end, uint32(-1) );
            tree.write_leaf( leaf_index, begin, end );
        }
    }
}

// do a single kd-split for all nodes in the input task queue, and generate
// a corresponding list of output tasks
template <typename Tree, typename Integer>
void split(
    Tree                tree,
    const uint32        max_leaf_size,
    const bool          keep_singletons,
    const Integer*      codes,
    const uint32        in_tasks_count,
    const Split_task*   in_tasks,
    const uint32*       in_skip_nodes,
    uint32*             out_tasks_count,
    Split_task*         out_tasks,
    uint32*             out_skip_nodes,
    const uint32        out_nodes_count,
    uint32*             out_leaf_count)
{
 //   const uint32 BLOCK_SIZE = 128;
	//const size_t max_blocks = 65535;
 //  // const size_t max_blocks = thrust::detail::backend::cuda::arch::max_active_blocks(split_kernel<BLOCK_SIZE,Tree,Integer>, BLOCK_SIZE, 0);
 //   const size_t n_blocks   = nih::min( max_blocks, (in_tasks_count + BLOCK_SIZE-1) / BLOCK_SIZE );
	const uint32 BLOCK_SIZE = 128;
	int numSMs;
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
	size_t max_blocks = 65535;
	const size_t n_blocks   = nih::min( max_blocks, (in_tasks_count + (BLOCK_SIZE*numSMs)-1) / (BLOCK_SIZE*numSMs) );
    const size_t grid_size  = n_blocks * BLOCK_SIZE*numSMs;

    split_kernel<BLOCK_SIZE> <<<n_blocks*numSMs,BLOCK_SIZE>>> (
        tree,
        max_leaf_size,
        keep_singletons,
        grid_size,
        codes,
        in_tasks_count,
        in_tasks,
        in_skip_nodes,
        out_tasks_count,
        out_tasks,
        out_skip_nodes,
        out_nodes_count,
        out_leaf_count );

    cudaThreadSynchronize();
}

// generate a leaf for each task
template <typename Tree>
void gen_leaves(
    Tree                tree,
    const uint32        in_tasks_count,
    const Split_task*   in_tasks,
    const uint32*       in_skip_nodes,
    uint32*             out_leaf_count)
{
    const uint32 BLOCK_SIZE = 128;
	const size_t max_blocks = BLOCK_SIZE;
    //const size_t max_blocks = thrust::detail::backend::cuda::arch::max_active_blocks(gen_leaves_kernel<Tree,BLOCK_SIZE>, BLOCK_SIZE, 0);
    const size_t n_blocks   = nih::min( max_blocks, (in_tasks_count + BLOCK_SIZE-1) / BLOCK_SIZE );
    const size_t grid_size  = n_blocks * BLOCK_SIZE;

    gen_leaves_kernel<Tree,BLOCK_SIZE> <<<n_blocks,BLOCK_SIZE>>> (
        tree,
        grid_size,
        in_tasks_count,
        in_tasks,
        in_skip_nodes,
        out_leaf_count );

    cudaThreadSynchronize();
}

} // namespace bintree

template <typename Tree, typename Integer>
void generate(
    Bintree_gen_context& context,
    const uint32    n_codes,
    const Integer*  codes,
    const uint32    bits,
    const uint32    max_leaf_size,
    const bool      keep_singletons,
    Tree&           tree)
{
    tree.reserve_nodes( n_codes * 2 );
    tree.reserve_leaves( n_codes );

    // start building the octree
    need_space( context.m_task_queues[0], n_codes );
    need_space( context.m_task_queues[1], n_codes );
    need_space( context.m_skip_nodes,     n_codes * 2 );

    Bintree_gen_context::Split_task* task_queues[2] = {
        thrust::raw_pointer_cast( &(context.m_task_queues[0]).front() ),
        thrust::raw_pointer_cast( &(context.m_task_queues[1]).front() )
    };
    uint32* skip_nodes[2] = {
        thrust::raw_pointer_cast( &(context.m_skip_nodes).front() ),
        thrust::raw_pointer_cast( &(context.m_skip_nodes).front() + n_codes )
    };

    uint32 in_queue  = 0;
    uint32 out_queue = 1;

    context.m_counters.resize( 4 );
    context.m_counters[ in_queue ]  = 1;
    context.m_counters[ out_queue ] = 0;
    context.m_counters[ 2 ]         = 0; // leaf counter

    context.m_task_queues[ in_queue ][0] = Bintree_gen_context::Split_task( 0, 0, n_codes, bits-1 );
    context.m_skip_nodes[0]              = uint32(-1);

    uint32 n_nodes = 1;

    // start splitting from the most significant bit
    int32 level = bits-1;

    context.m_levels[ bits ] = 0;

    // loop until there's tasks left in the input queue
    while (context.m_counters[ in_queue ] && level >= 0)
    {
        context.m_levels[ level ] = n_nodes;

        tree.reserve_nodes( n_nodes + context.m_counters[ in_queue ]*2 );

        // clear the output queue
        context.m_counters[ out_queue ] = 0;
        cudaThreadSynchronize();

        bintree::split(
            tree.get_context(),
            max_leaf_size,
            keep_singletons,
            codes,
            context.m_counters[ in_queue ],
            task_queues[ in_queue ],
            skip_nodes[ in_queue ],
            thrust::raw_pointer_cast( &context.m_counters.front() ) + out_queue,
            task_queues[ out_queue ],
            skip_nodes[ out_queue ],
            n_nodes,
            thrust::raw_pointer_cast( &context.m_counters.front() ) + 2 );

        const uint32 out_count = context.m_counters[ out_queue ];

        // update the number of nodes
        n_nodes += out_count;

        // swap the input and output queues
        std::swap( in_queue, out_queue );

        // decrease the level
        --level;
    }

    for (; level >= 0; --level)
        context.m_levels[ level ] = n_nodes;

    // generate a leaf for each of the remaining tasks
    if (context.m_counters[ in_queue ])
    {
        bintree::gen_leaves(
            tree.get_context(),
            context.m_counters[ in_queue ],
            task_queues[ in_queue ],
            skip_nodes[ in_queue ],
            thrust::raw_pointer_cast( &context.m_counters.front() ) + 2 );
    }
    context.m_nodes  = n_nodes;
    context.m_leaves = context.m_counters[2];
}

} // namespace cuda
} // namespace nih
