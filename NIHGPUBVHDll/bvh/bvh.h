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

/*! \file bvh.h
 *   \brief Entry point to the generic Bounding Volume Hierarchy library.
 */

#pragma once

#include "../linalg/vector.h"
#include "../linalg/bbox.h"
#include <vector>
#include <stack>

namespace nih {

///
/// Bvh node struct
///
struct Bvh_node
{
    typedef uint32 Type;
    const static uint32 kLeaf     = (1u << 31u);
    const static uint32 kInternal = 0x00000000u;
    const static uint32 kInvalid  = uint32(-1);

    /// empty constructor
    ///
    NIH_HOST NIH_DEVICE Bvh_node() {}
    /// full constructor
    ///
    /// \param type         node type
    /// \param index        child/leaf index
    /// \param skip_node    skip node index
    NIH_HOST NIH_DEVICE Bvh_node(const Type type, const uint32 index, const uint32 skip_node);
    /// set node type
    ///
    /// \param type         node type
	NIH_HOST NIH_DEVICE void set_type(const Type type);
    /// set child/leaf index
    ///
    /// \param index        child/leaf index
	NIH_HOST NIH_DEVICE void set_index(const uint32 index);
    /// set skip node index
    ///
    /// \param index        skip node index
    NIH_HOST NIH_DEVICE void set_skip_node(const uint32 index);

    /// is a leaf?
    ///
    NIH_HOST NIH_DEVICE bool is_leaf() const { return (m_packed_data & kLeaf) != 0u; }
    /// get child/leaf index
    ///
    NIH_HOST NIH_DEVICE uint32 get_index() const { return m_packed_data & (~kLeaf); }
    /// get leaf index
    ///
    NIH_HOST NIH_DEVICE uint32 get_leaf_index() const { return m_packed_data & (~kLeaf); }
    /// get skip node index
    ///
    NIH_HOST NIH_DEVICE uint32 get_skip_node() const { return m_skip_node; }

    /// get child count
    ///
    NIH_HOST NIH_DEVICE uint32 get_child_count() const { return 2u; }
    /// get i-th child
    ///
    /// \param i    child index
    NIH_HOST NIH_DEVICE uint32 get_child(const uint32 i) const { return get_index() + i; }

    /// compute packed data
    ///
    /// \param type     node type
    /// \param index    child/leaf index
    static NIH_HOST NIH_DEVICE uint32 packed_data(const Type type, const uint32 index)
    {
	    return (uint32(type) | index);
    }
    /// set node type into a packed data
    ///
    /// \param packed_data  packed data
    /// \param type         node type
    static NIH_HOST NIH_DEVICE void set_type(uint32& packed_data, const Type type)
    {
	    packed_data &= ~kLeaf;
	    packed_data |= uint32(type);
    }
    /// set child/leaf index into a packed data
    ///
    /// \param packed_data  packed data
    /// \param index        child/leaf index
    static NIH_HOST NIH_DEVICE void set_index(uint32& packed_data, const uint32 index)
    {
	    packed_data &= kLeaf;
	    packed_data |= index;
    }
    static NIH_HOST NIH_DEVICE bool   is_leaf(const uint32 packed_data)   { return (packed_data & kLeaf) != 0u; }
    static NIH_HOST NIH_DEVICE uint32 get_index(const uint32 packed_data) { return packed_data & (~kLeaf); }

	uint32	m_packed_data;	// child index
	uint32	m_skip_node;	// skip node index
};

///
/// Bvh leaf struct
///
struct Bvh_leaf
{
    /// empty constructor
    ///
	NIH_HOST NIH_DEVICE Bvh_leaf() {}
    /// constructor
    ///
    /// \param leaf size
    /// \param first element
	NIH_HOST NIH_DEVICE Bvh_leaf(
		const uint32 size,
		const uint32 index) :
		m_size( size ), m_index( index ) {}

    /// get leaf size
    ///
	NIH_HOST NIH_DEVICE uint32 get_size() const { return m_size; }
    /// get first element
    ///
	NIH_HOST NIH_DEVICE uint32 get_index() const { return m_index; }

	uint32 m_size;
	uint32 m_index;
};

///
/// A low dimensional bvh class
///
template <uint32 DIM>
struct Bvh
{
	typedef Vector<float,DIM>	Vector_type;
	typedef Bbox<Vector_type>	Bbox_type;

	typedef Bvh_node			Node_type;
	typedef Bvh_leaf			Leaf_type;

	std::vector<Bvh_node>		m_nodes;
	std::vector<Bvh_leaf>		m_leaves;
	std::vector<Bbox_type>		m_bboxes;
};

///
/// A bvh builder for sets of low dimensional bboxes
///
template <uint32 DIM>
class Bvh_builder
{
public:
	typedef Vector<float,DIM>	Vector_type;
	typedef Bbox<Vector_type>	Bbox_type;

	/// constructor
    ///
	Bvh_builder() : m_max_leaf_size( 4u ) {}

	/// set bvh parameters
    ///
    /// \param max_leaf_size    maximum leaf size
	void set_params(const uint32 max_leaf_size) { m_max_leaf_size = max_leaf_size; }

	/// build
	///
	/// Iterator is supposed to dereference to a Vector<float,DIM>
	///
	/// \param begin			first point
	/// \param end				last point
	/// \param bvh				output bvh
	template <typename Iterator>
	void build(
		const Iterator	begin,
		const Iterator	end,
		Bvh<DIM>*		bvh);

	/// remapped point index
    ///
	uint32 index(const uint32 i) { return m_points[i].m_index; }

private:
	struct Point
	{
		Bbox_type	m_bbox;
		uint32		m_index;

        float center(const uint32 dim) const { return (m_bbox[0][dim] + m_bbox[1][dim])*0.5f; }
	};

	struct Node
	{
		uint32		m_begin;
		uint32		m_end;
		uint32		m_node;
		uint32		m_depth;
	};
	typedef std::stack<Node> Node_stack;

	void compute_bbox(
		const uint32		begin,
		const uint32		end,
		Bbox_type&			bbox);

	struct Bvh_partitioner;

	uint32				m_max_leaf_size;
	std::vector<Point>	m_points;
};

/// compute SAH cost of a subtree
inline float compute_sah_cost(const Bvh<3>& bvh, uint32 node_index = 0);

} // namespace nih

#include "../bvh/bvh_inline.h"
