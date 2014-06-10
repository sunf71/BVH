#pragma once
#include "../basic/types.h"
#include "../linalg/vector.h"
#include "../linalg/bbox.h"
namespace nih
{
	//Frustum plane
	typedef struct plane
	{
		float a;
		float b;
		float c;
		float d;
	} plane_t;
	//

	//Pyramidal Frustum
	typedef struct pyrfrustum
	{
		plane_t planes[6];
	} pyrfrustum_t;

	//Pyramidal Frustum Corners
	typedef struct pyrcorners
	{
		Vector3f points[8];
	} pyrcorners_t;

	FORCE_INLINE NIH_HOST_DEVICE void NormalizePlane(plane_t & plane)
	{
		float mag;
		mag = sqrt(plane.a * plane.a + plane.b * plane.b + plane.c * plane.c);
		plane.a = plane.a / mag;
		plane.b = plane.b / mag;
		plane.c = plane.c / mag;
		plane.d = plane.d / mag;
	}
	FORCE_INLINE NIH_HOST_DEVICE float planeDistance( Vector3f & v, plane_t& p )
	{
		return (v[0]*p.a + v[1]*p.b + v[2]*p.c + p.d );
	}

	FORCE_INLINE NIH_HOST_DEVICE bool AABBcontainsPoint( Bbox4f& a, Vector3f& p )
	{
		/*return (p[0] >= a.m_min[0] && p[0] <= a.m_max[0]) &&
			(p[1] >= a.m_min[1] && p[1] <= a.m_max[1]) &&
			(p[2] >= a.m_min[2] && p[2] <= a.m_max[2]);*/
		for( size_t i=0; i<4; i++)
		{
			if ( p[i] < a.m_min[i] || p[i] > a.m_max[i])
				return false;
		}
		return true;
	}


	FORCE_INLINE NIH_HOST_DEVICE bool AABBenclosing( Bbox4f& a, pyrcorners_t& c )
	{
		
		for( int i = 0; i < 8; i++ )
			if( AABBcontainsPoint( a, c.points[i] ) )
				return true;
		return false;
	}

	FORCE_INLINE NIH_HOST_DEVICE bool Intersect( pyrfrustum_t& f, Bbox4f& a )
	{
		
		//if( AABBenclosing( a, c ) )
		//	return true;

		Vector3f box[8];
		box[0][0] = a.m_min[0]; box[0][1] = a.m_min[1]; box[0][2] = a.m_min[2];
		box[1][0] = a.m_max[0]; box[1][1] = a.m_min[1]; box[1][2] = a.m_min[2];
		box[2][0] = a.m_min[0]; box[2][1] = a.m_max[1]; box[2][2] = a.m_min[2];
		box[3][0] = a.m_max[0]; box[3][1] = a.m_max[1]; box[3][2] = a.m_min[2];
		box[4][0] = a.m_min[0]; box[4][1] = a.m_min[1]; box[4][2] = a.m_max[2];
		box[5][0] = a.m_max[0]; box[5][1] = a.m_min[1]; box[5][2] = a.m_max[2];
		box[6][0] = a.m_min[0]; box[6][1] = a.m_max[1]; box[6][2] = a.m_max[2];
		box[7][0] = a.m_max[0]; box[7][1] = a.m_max[1]; box[7][2] = a.m_max[2];
		
		int iTotalIn = 0;

		// test all 8 corners against the 6 sides 
		// if all points are behind 1 specific plane, we are out
		// if we are in with all points, then we are fully in
		for(int p = 0; p < 6; ++p) {

			int iInCount = 8;
			int iPtIn = 1;

			for(int i = 0; i < 8; ++i) {

				// test this point against the planes
				if(planeDistance( box[i], f.planes[p] ) <= 0 ) {
					iPtIn = 0;
					--iInCount;
				}
			}

			// were all the points outside of plane p?
			if(iInCount == 0)
				return false;

			// check if they were all on the right side of the plane
			iTotalIn += iPtIn;
		}

		// so if iTotalIn is 6, then all are inside the view
		if(iTotalIn == 6)
			return true;

		// we must be partly in then otherwise
		return true;
	}
	

	FORCE_INLINE NIH_HOST_DEVICE bool FrustumContainPoints(pyrfrustum_t& f, Vector3f & o )
	{
		for(int p = 0; p < 6; ++p) {
			if(planeDistance( o, f.planes[p] ) <= 0 ) {
				return false;
			}
			
		}
		return true;

			
	}
}


