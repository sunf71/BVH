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

#include <iterator>

namespace nih {

/*! \addtogroup trees Trees
 *  \{
 */

struct breadth_first_tree {};
struct depth_first_tree   {};

/// A simple Breadth-First Tree model implementation
template <typename Node_iterator, typename Domain_type>
struct BFTree
{
    typedef Domain_type         domain_type;
    typedef Node_iterator       node_iterator;
    typedef typename std::iterator_traits<node_iterator>::value_type node_type;
    typedef breadth_first_tree  tree_type;

    /// empty constructor
    BFTree() {}

    /// constructor
    BFTree(
        const node_iterator nodes,
        const uint32        n_leaves,
        const uint2*        leaves,
        const uint32        n_levels,
        const uint32*       levels) : 
        m_nodes( nodes ),
        m_leaf_count( n_leaves ),
        m_leaves( leaves ),
        m_level_count( n_levels ),
        m_levels( levels )
    {}

    /// return the number of levels
    NIH_HOST_DEVICE uint32 get_level_count() const { return m_level_count; }

    /// return the i-th level
    NIH_HOST_DEVICE uint32 get_level(const uint32 i) const { return m_levels[i]; }

    /// retrieve a node
    NIH_HOST_DEVICE node_type get_node(const uint32 index) const
    {
        return m_nodes[ index ];
    }

    /// return the number of leaves
    NIH_HOST_DEVICE uint32 get_leaf_count() const { return m_leaf_count; }

    /// retrieve a leaf
    NIH_HOST_DEVICE uint2 get_leaf(const uint32 index) const
    {
        return m_leaves[ index ];
    }

    const node_iterator m_nodes;
    uint32              m_leaf_count;
    const uint2*        m_leaves;
    uint32              m_level_count;
    const uint32*       m_levels;
};

/*! \}
 */

} // namespace nih
