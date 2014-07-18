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

/*! \file bbox.h
 *   \brief Defines an axis-aligned bounding box class.
 */

#pragma once

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#include "numbers.h"
#include "vector.h"
#include <limits>
#include <algorithm>
#include <iostream>
namespace nih {

/*! \addtogroup linalg Linear Algebra
 */

/*! \addtogroup bboxes Bounding Boxes
 *  \ingroup linalg
 *  \{
 */

///
/// Axis-Aligned Bounding Bbox class, templated over an arbitrary vector type
///
template <typename Vector_t>
struct Bbox
{
	typedef typename Vector_t::Field_type	Field_type;
	typedef typename Vector_t				Vector_type;

    /// empty constructor
    ///
	NIH_HOST NIH_DEVICE Bbox();

    /// point constructor
    ///
    /// \param v    point
	NIH_HOST NIH_DEVICE Bbox(
		const Vector_t& v);

    /// min/max constructor
    ///
    /// \param v1   min corner
    /// \param v2   max corner
    NIH_HOST NIH_DEVICE Bbox(
		const Vector_t& v1,
		const Vector_t& v2);

    /// merging constructor
    ///
    /// \param bb1  first bbox
    /// \param bb2  second bbox
	NIH_HOST NIH_DEVICE Bbox(
		const Bbox<Vector_t>& bb1,
		const Bbox<Vector_t>& bb2);

    /// copy constructor
    ///
    /// \param bb   bbox to copy
    NIH_HOST NIH_DEVICE Bbox(
		const Bbox<Vector_t>& bb);

    /// insert a point
    ///
    /// \param v    point to insert
	NIH_HOST NIH_DEVICE void insert(const Vector_t& v);

    /// insert a bbox
    ///
    /// \param v    bbox to insert
	NIH_HOST NIH_DEVICE void insert(const Bbox& v);

    /// clear bbox
    ///
	NIH_HOST NIH_DEVICE void clear();

    /// const corner indexing operator
    ///
    /// \param i    corner to retrieve
	NIH_HOST NIH_DEVICE const Vector_t& operator[](const size_t i) const	{ return (&m_min)[i]; }

    /// corner indexing operator
    ///
    /// \param i    corner to retrieve
	NIH_HOST NIH_DEVICE Vector_t& operator[](const size_t i)				{ return (&m_min)[i]; }

    /// copy operator
    ///
    /// \param bb   bbox to copy
    NIH_HOST NIH_DEVICE Bbox<Vector_t>& operator=(const Bbox<Vector_t>& bb);

    Vector_t m_min; ///< min corner
	Vector_t m_max; ///< max corner
};

typedef Bbox<Vector2f> Bbox2f;
typedef Bbox<Vector3f> Bbox3f;
typedef Bbox<Vector4f> Bbox4f;
typedef Bbox<Vector2d> Bbox2d;
typedef Bbox<Vector3d> Bbox3d;
typedef Bbox<Vector4d> Bbox4d;

/// compute the area of a 3d bbox
///
/// \param bbox     bbox object
inline NIH_HOST_DEVICE float area(const Bbox3f& bbox);

/// point-in-bbox inclusion predicate
///
/// \param bbox     bbox object
/// \param p        point to test for inclusion
template <typename Vector_t>
inline NIH_HOST_DEVICE bool contains(const Bbox<Vector_t>& bbox, const Vector_t& p);

/// point-to-bbox squared distance
///
/// \param bbox     bbox object
/// \param p        point
template <typename Vector_t>
FORCE_INLINE NIH_HOST_DEVICE float sq_distance(const Bbox<Vector_t>& bbox, const Vector_t& p);

/// returns the largest axis of a bbox
///
/// \param bbox     bbox object
template <typename Vector_t>
FORCE_INLINE NIH_HOST_DEVICE size_t largest_axis(const Bbox<Vector_t>& bbox);

/*! \}
 */
template <typename Vector_t>
struct Add_Bbox {
   FORCE_INLINE  NIH_HOST_DEVICE Bbox<Vector_t> operator()(const Bbox<Vector_t>& a, const Bbox<Vector_t>& b) const ;
};

template <typename Vector_t>
    FORCE_INLINE  NIH_HOST_DEVICE bool BoxesIntersect(const Bbox<Vector_t>& a, const Bbox<Vector_t>& b);

} // namespace nih

#include "bbox_inline.h"
