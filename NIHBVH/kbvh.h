#pragma once
#include "bvh\bvh.h"


class KBvh_builder
{
public:
	typedef Vector<float,DIM>	Vector_type;
	typedef Bbox<Vector_type>	Bbox_type;

	/// constructor
    ///
	KBvh_builder() : m_max_leaf_size( 4u ) {}

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
		const Bbox3f    bbox,
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

	uint32				m_max_leaf_size;
	std::vector<Point>	m_points;
	std::vector<DIM> m_codes;
};
