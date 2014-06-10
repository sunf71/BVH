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

/*! \file bintree_node.h
 *   \brief Define CUDA based scan primitives.
 */

#pragma once

#include "../basic/types.h"

namespace nih {

/*! \addtogroup bintree Binary Trees
 *  \{
 */

///
/// A middle-split binary tree node.
/// A node can either be a leaf and have no children, or be
/// an internal split node. If a split node, it can either
/// have one or two children: for example, it can have one
/// if a set of points is concentrated in one half-space.
///
struct Bintree_node
{
    static const uint32 kInvalid = uint32(-1);

    /// empty constructor
    ///
    NIH_HOST_DEVICE Bintree_node() {}

    /// full constructor
    ///
    /// \param child0   first child activation predicate
    /// \param child1   second child activation predicate
    /// \param index    child index
    NIH_HOST_DEVICE Bintree_node(bool child0, bool child1, uint32 index) :
        m_packed_info( (child0 ? 1u : 0u) | (child1 ? 2u : 0u) | (index << 2) ) {}

    /// is a leaf?
    ///
    NIH_HOST_DEVICE uint32 is_leaf() const
    {
        return (m_packed_info & 3u) == 0u;
    }
    /// get offset of the first child
    ///
    NIH_HOST_DEVICE uint32 get_child_offset() const
    {
        return m_packed_info >> 2u;
    }
    /// get leaf index
    ///
    NIH_HOST_DEVICE uint32 get_leaf_index() const
    {
        return m_packed_info >> 2u;
    }
    /// get i-th child (among the active ones)
    ///
    /// \param i    child index
    NIH_HOST_DEVICE uint32 get_child(const uint32 i) const
    {
        return get_child_offset() + i;
    }
    /// is the i-th child active?
    ///
    /// \param i    child index
    NIH_HOST_DEVICE bool has_child(const uint32 i) const
    {
        return m_packed_info & (1u << i) ? true : false;
    }
    /// get left partition (or kInvalid if not active)
    ///
    NIH_HOST_DEVICE uint32 get_left() const
    {
        return has_child(0) ? get_child_offset() : kInvalid;
    }
    /// get right partition (or kInvalid if not active)
    ///
    NIH_HOST_DEVICE uint32 get_right() const
    {
        return has_child(1) ? get_child_offset() + (has_child(0) ? 1u : 0u) : kInvalid;
    }

    uint32 m_packed_info;
};

/*! \}
 */

} // namespace nih
