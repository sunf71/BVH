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
#include "lbvh_context.h"
#include "../../bvh/cuda/lbvh_builder.h"
#include "../../bintree/bintree_gen.h"
#include "../../sampling/random.h"
#include "../../time/timer.h"
#include "../../basic/cuda_domains.h"
#include "../../tree/model.h"
#include "../../tree/cuda/reduce.h"

namespace nih {

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

	/// A simple binary tree context implementation to be used with
/// the Bvh generate() function.
struct LBVH_context
{
    /// Cuda accessor struct
    struct Context
    {
        NIH_HOST_DEVICE Context() {}
        NIH_HOST_DEVICE Context(Bvh_node* nodes, uint2* leaves) :
            m_nodes(nodes), m_leaves(leaves) {}

        /// write a new node
        NIH_HOST_DEVICE void write_node(const uint32 node, bool p1, bool p2, const uint32 offset, const uint32 skip_node, const uint32 level, const uint32 begin, const uint32 end, const uint32 split_index)
        {
            const uint32 type = p1 == false && p2 == false ? Bvh_node::kLeaf : Bvh_node::kInternal;
            m_nodes[ node ] = Bvh_node( type, offset, skip_node );
        }
        /// write a new leaf
        NIH_HOST_DEVICE void write_leaf(const uint32 index, const uint32 begin, const uint32 end)
        {
            m_leaves[ index ] = make_uint2( begin, end );
        }

        Bvh_node*  m_nodes;    ///< node pointer
        uint2*     m_leaves;   ///< leaf pointer
    };

    /// constructor
    LBVH_context(
        thrust::host_vector<Bvh_node>* nodes,
        thrust::host_vector<uint2>*    leaves) :
        m_nodes( nodes ), m_leaves( leaves ) {}

        /// reserve space for more nodes
    void reserve_nodes(const uint32 n) { if (m_nodes->size() < n) m_nodes->resize(n); }

    /// reserve space for more leaves
    void reserve_leaves(const uint32 n) { if (m_leaves->size() < n) m_leaves->resize(n); }

    /// return a cuda context
    Context get_context()
    {
        return Context(
            thrust::raw_pointer_cast( &m_nodes->front() ),
            thrust::raw_pointer_cast( &m_leaves->front() ) );
    }

    thrust::host_vector<Bvh_node>* m_nodes;
    thrust::host_vector<uint2>*    m_leaves;
};
void lbvh_test()
{
    fprintf(stderr, "lbvh test... started\n");

    const uint32 n_points = 837*1024;
    const uint32 n_tests = 100;

    thrust::host_vector<Vector4f> h_points( n_points );

    Random random;
    for (uint32 i = 0; i < n_points; ++i)
        h_points[i] = Vector4f( random.next(), random.next(), random.next(), 1.0f );

    thrust::device_vector<Vector4f> d_points( h_points );
    thrust::device_vector<Vector4f> d_unsorted_points( h_points );

    thrust::device_vector<Bvh_node> bvh_nodes;
    thrust::device_vector<uint2>    bvh_leaves;
    thrust::device_vector<uint32>   bvh_index;

    cuda::LBVH_builder<uint64> builder( bvh_nodes, bvh_leaves, bvh_index );

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    float time = 0.0f;

    for (uint32 i = 0; i <= n_tests; ++i)
    {
        d_points = d_unsorted_points;
        cudaThreadSynchronize();

        float dtime;
        cudaEventRecord( start, 0 );

        builder.build(
            Bbox3f( Vector3f(0.0f), Vector3f(1.0f) ),
            d_points.begin(),
            d_points.end(),
            16u );

        cudaEventRecord( stop, 0 );
        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &dtime, start, stop );

        if (i) // skip the first run
            time += dtime;
    }
    time /= 1000.0f * float(n_tests);

    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    {
        thrust::host_vector<uint64>   h_codes( builder.m_codes );
        thrust::host_vector<Bvh_node> h_nodes;
        thrust::host_vector<uint2>    h_leaves;

        const uint32 n_codes = n_points;

        LBVH_context tree( &h_nodes, &h_leaves );
        generate(
            n_codes,
            &h_codes[0],
            60,
            16u,
            false,
            tree );

        thrust::host_vector<Bvh_node> d_nodes( bvh_nodes );
        thrust::host_vector<uint2>    d_leaves( bvh_leaves );

        // traverse both trees top-down to see whether there's any inconsistencies...
        uint32 h_node_id = 0;
        uint32 d_node_id = 0;
        uint32 node_index = 0;
        uint32 leaf_index = 0;

        while (h_node_id != uint32(-1))
        {
            if (d_node_id == uint32(-1))
            {
                fprintf(stderr, "device node is invalid!\n");
                break;
            }

            Bvh_node h_node = h_nodes[ h_node_id ];
            Bvh_node d_node = d_nodes[ d_node_id ];

            if (h_node.is_leaf() != d_node.is_leaf())
            {
                fprintf(stderr, "host node and device node have different topology! (%u) (%s, %s)\n", node_index, h_node.is_leaf() ? "leaf" : "split", d_node.is_leaf() ? "leaf" : "split" );
                break;
            }

            if (h_node.is_leaf())
            {
                const uint2 h_leaf = h_leaves[ h_node.get_leaf_index() ];
                const uint2 d_leaf = d_leaves[ d_node.get_leaf_index() ];

                if (h_leaf.x != d_leaf.x ||
                    h_leaf.y != d_leaf.y)
                {
                    fprintf(stderr, "host and device leaves differ! [%u,%u) != [%u,%u) (%u:%u)\n",
                        h_leaf.x, h_leaf.y,
                        d_leaf.x, d_leaf.y,
                        node_index, leaf_index );
                    break;
                }

                h_node_id = h_node.get_skip_node();
                d_node_id = d_node.get_skip_node();

                leaf_index++;
            }
            else
            {
                h_node_id = h_node.get_child(0);
                d_node_id = d_node.get_child(0);
            }

            node_index++;
        }
   }

    fprintf(stderr, "lbvh test... done\n");
    fprintf(stderr, "  time       : %f ms\n", time * 1000.0f );
    fprintf(stderr, "  points/sec : %f M\n", (n_points / time) / 1.0e6f );

    fprintf(stderr, "  nodes  : %u\n", builder.m_node_count );
    fprintf(stderr, "  leaves : %u\n", builder.m_leaf_count );
    for (uint32 level = 0; level < 60; ++level)
        fprintf(stderr, "  level %u : %u nodes\n", level, builder.m_levels[level+1] - builder.m_levels[level] );

    fprintf(stderr, "lbvh bbox reduction test... started\n");

    BFTree<Bvh_node*,device_domain> bvh(
        thrust::raw_pointer_cast( &bvh_nodes.front() ),
        builder.m_leaf_count,
        thrust::raw_pointer_cast( &bvh_leaves.front() ),
        60u,
        builder.m_levels );

    thrust::device_vector<Bbox4f> d_leaf_bboxes( builder.m_leaf_count );
    thrust::device_vector<Bbox4f> d_node_bboxes( builder.m_node_count );

    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    time = 0.0f;

    for (uint32 i = 0; i <= n_tests; ++i)
    {
        float dtime;
        cudaEventRecord( start, 0 );

        cuda::tree_reduce(
            bvh,
            thrust::raw_pointer_cast( &d_points.front() ),
            thrust::raw_pointer_cast( &d_leaf_bboxes.front() ),
            thrust::raw_pointer_cast( &d_node_bboxes.front() ),
            bbox_functor(),
            Bbox4f() );

        cudaEventRecord( stop, 0 );
        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &dtime, start, stop );

        if (i) // skip the first run
            time += dtime;
    }
    time /= 1000.0f * float(n_tests);

    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    fprintf(stderr, "lbvh bbox reduction test... done\n");
    fprintf(stderr, "  time       : %f ms\n", time * 1000.0f );
    fprintf(stderr, "  points/sec : %f M\n", (n_points / time) / 1.0e6f );
}

} // namespace nih

