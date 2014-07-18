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

namespace nih {

	template <typename Vector_t>
	Bbox<Vector_t>::Bbox() :
	m_min( Field_traits<Field_type>::max() ),
		m_max( Field_traits<Field_type>::min() )
	{
	}
	template <typename Vector_t>
	Bbox<Vector_t>::Bbox(const Vector_t& v) :
	m_min( v ),
		m_max( v )
	{
	}
	template <typename Vector_t>
	Bbox<Vector_t>::Bbox(const Vector_t& v1, const Vector_t& v2) :
	m_min( v1 ),
		m_max( v2 )
	{
	}
	template <typename Vector_t>
	Bbox<Vector_t>::Bbox(const Bbox<Vector_t>& bb1, const Bbox<Vector_t>& bb2)
	{
		for (size_t i = 0; i < m_min.dimension(); i++)
		{
			m_min[i] = ::nih::min( bb1[0][i], bb2[0][i] );
			m_max[i] = ::nih::max( bb1[1][i], bb2[1][i] );
		}
	}
	template <typename Vector_t>
	Bbox<Vector_t>::Bbox(const Bbox<Vector_t>& bb) :
	m_min( bb.m_min ),
		m_max( bb.m_max )
	{
	}

	template <typename Vector_t>
	void Bbox<Vector_t>::insert(
		const Vector_t& v)
	{
		for (size_t i = 0; i < m_min.dimension(); i++)
		{
			m_min[i] = ::nih::min( m_min[i], v[i] );
			m_max[i] = ::nih::max( m_max[i], v[i] );
		}
	}
	template <typename Vector_t>
	void Bbox<Vector_t>::insert(
		const Bbox& bbox)
	{
		for (size_t i = 0; i < m_min.dimension(); i++)
		{
			m_min[i] = ::nih::min( m_min[i], bbox.m_min[i] );
			m_max[i] = ::nih::max( m_max[i], bbox.m_max[i] );
		}
	}
	template <typename Vector_t>
	void Bbox<Vector_t>::clear()
	{
		for (size_t i = 0; i < m_min.dimension(); i++)
		{
			m_min[i] = Field_traits<Field_type>::max();
			m_max[i] = Field_traits<Field_type>::min();
		}
	}

	template <typename Vector_t>
	Bbox<Vector_t>& Bbox<Vector_t>::operator=(const Bbox<Vector_t>& bb)
	{
		m_min = bb.m_min;
		m_max = bb.m_max;
		return *this;
	}

	template <typename Vector_t>
	size_t largest_axis(const Bbox<Vector_t>& bbox)
	{
		typedef typename Vector_t::Field_type Field_type;

		const Vector_t edge( bbox[1] - bbox[0] );
		size_t axis = 0;
		Field_type v = edge[0];

		for (size_t i = 1; i < edge.dimension(); i++)
		{
			if (v < edge[i])
			{
				v = edge[i];
				axis = i;
			}
		}
		return axis;
	}

	// compute the area of a 3d bbox
	//
	// \param bbox     bbox object
	float area(const Bbox3f& bbox)
	{
		const Vector3f edge = bbox[1] - bbox[0];
		return edge[0] * edge[1] + edge[2] * (edge[0] + edge[1]);
	}

	// point-in-bbox inclusion predicate
	//
	// \param bbox     bbox object
	// \param p        point to test for inclusion
	template <typename Vector_t>
	bool contains(const Bbox<Vector_t>& bbox, const Vector_t& p)
	{
		for (uint32 i = 0; i < p.dimension(); ++i)
		{
			if (p[i] < bbox[0][i] ||
				p[i] > bbox[1][i])
				return false;
		}
		return true;
	}

	// point-to-bbox squared distance
	//
	// \param bbox     bbox object
	// \param p        point
	template <typename Vector_t>
	float sq_distance(const Bbox<Vector_t>& bbox, const Vector_t& p)
	{
		float r = 0.0f;
		for (uint32 i = 0; i < p.dimension(); ++i)
		{
			const float dist = nih::max( bbox[0][i] - p[i], 0.0f ) +
				nih::max( p[i] - bbox[1][i], 0.0f );

			r += dist*dist;
		}
		return r;
	}
	inline void printVector3f(const Vector3f& point)
	{
		std::cout<<point[0]<<","<<point[1]<<","<<point[2]<<std::endl;
	}
	inline void printBbox3f(const Bbox3f& bbox)
	{
		std::cout<<"min "<<bbox.m_min[0]<<","<<bbox.m_min[1]<<","<<bbox.m_min[2];
		std::cout<<" max "<<bbox.m_max[0]<<","<<bbox.m_max[1]<<","<<bbox.m_max[2]<<std::endl;
	}
	template <typename Vector_t>
	inline Bbox<Vector_t> Add_Bbox<Vector_t>::operator()(const Bbox<Vector_t>& a, const Bbox<Vector_t>& b) const
	{
        Bbox<Vector_t> r;
		r.insert(a);
		r.insert(b);
        return r;
    }
	template <typename Vector_t>
	bool BoxesIntersect(const Bbox<Vector_t>& a, const Bbox<Vector_t>& b) 
	{
		return (a.m_max[0] > b.m_min[0]&&
			a.m_min[0] < b.m_max[0] &&
			a.m_max[1] > b.m_min[1] &&
			a.m_min[1] < b.m_max[1] &&
			a.m_max[2] > b.m_min[2] &&
			a.m_min[2] < b.m_max[2]);
	}
} // namespace nih