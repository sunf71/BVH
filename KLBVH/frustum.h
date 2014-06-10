#pragma once
#include "types.h"
#include "vector.h"
#include "bbox.h"
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

	//0 out
	//1 in
	//2 intersect
	FORCE_INLINE NIH_HOST_DEVICE int Intersect( pyrfrustum_t& f, Bbox3f& a )
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
				return 0;

			// check if they were all on the right side of the plane
			iTotalIn += iPtIn;
		}

		// so if iTotalIn is 6, then all are inside the view
		if(iTotalIn == 6)
			return 1;

		// we must be partly in then otherwise
		return 2;
	}
	

	/*FORCE_INLINE NIH_HOST_DEVICE Vector3f getVertexP(Bbox3f& box, Vector3f& normal)
	{

	}*/
	//@return:
	//0 out
	//1 in
	//2 intersect
	FORCE_INLINE NIH_HOST_DEVICE int IntersectFast( pyrfrustum_t& f, Bbox3f& a )
	{

		bool intersec = false;
		uint32 tableXX[2] = {3,0};
		uint32 tableYY[2] = {4,1};
		uint32 tableZZ[2] = {5,2};
		uint32 tableX[2] = {0,3};
		uint32 tableY[2] = {1,4};
		uint32 tableZ[2] = {2,5};
		float*p = &a.m_min[0];
		for(int i=0; i<6; i++)
		{
			//plane_t plane= f.planes[i];
			uint32 sa = f.planes[i].a > 0; 
			uint32 sb = f.planes[i].b >0; 
			uint32 sc = f.planes[i].c > 0;
			if (p[tableX[sa]]*f.planes[i].a + p[tableY[sb]]*f.planes[i].b + p[tableZ[sc]]*f.planes[i].c+f.planes[i].d <=0)
				return 0;
			if (p[tableXX[sa]]*f.planes[i].a + p[tableYY[sb]]*f.planes[i].b + p[tableZZ[sc]]*f.planes[i].c+f.planes[i].d <=0)
				intersec = true;

		}
		return intersec ? 2 : 1; 

	}

	FORCE_INLINE NIH_HOST_DEVICE int Intersect( pyrfrustum_t& f, float*ptr )
	{

		Vector3f box[8];
		box[0][0] = ptr[0]; box[0][1] = ptr[1]; box[0][2] = ptr[2];
		box[1][0] = ptr[3]; box[1][1] = ptr[1]; box[1][2] = ptr[2];
		box[2][0] = ptr[0]; box[2][1] = ptr[4]; box[2][2] = ptr[2];
		box[3][0] = ptr[3]; box[3][1] = ptr[4]; box[3][2] = ptr[2];
		box[4][0] = ptr[0]; box[4][1] = ptr[1]; box[4][2] = ptr[5];
		box[5][0] = ptr[3]; box[5][1] = ptr[1]; box[5][2] = ptr[5];
		box[6][0] = ptr[0]; box[6][1] = ptr[4]; box[6][2] = ptr[5];
		box[7][0] = ptr[3]; box[7][1] = ptr[4]; box[7][2] = ptr[5];
		
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
				return 0;

			// check if they were all on the right side of the plane
			iTotalIn += iPtIn;
		}

		// so if iTotalIn is 6, then all are inside the view
		if(iTotalIn == 6)
			return 1;

		// we must be partly in then otherwise
		return 2;
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

	struct Matrix4x4
	{
		// The elements of the 4x4 matrix are stored in
		// column-major order (see "OpenGL Programming Guide",
		// 3rd edition, pp 106, glLoadMatrix).
		float _11, _21, _31, _41;
		float _12, _22, _32, _42;
		float _13, _23, _33, _43;
		float _14, _24, _34, _44;
	};
	FORCE_INLINE NIH_HOST_DEVICE void ExtractPlanesGL(
		plane_t * p_planes,
		const Matrix4x4 & comboMatrix,
		bool normalize)
	{
		// Left clipping plane
		p_planes[0].a = comboMatrix._41 + comboMatrix._11;
		p_planes[0].b = comboMatrix._42 + comboMatrix._12;
		p_planes[0].c = comboMatrix._43 + comboMatrix._13;
		p_planes[0].d = comboMatrix._44 + comboMatrix._14;
		// Right clipping plane
		p_planes[1].a = comboMatrix._41 - comboMatrix._11;
		p_planes[1].b = comboMatrix._42 - comboMatrix._12;
		p_planes[1].c = comboMatrix._43 - comboMatrix._13;
		p_planes[1].d = comboMatrix._44 - comboMatrix._14;
		// Top clipping plane
		p_planes[2].a = comboMatrix._41 - comboMatrix._21;
		p_planes[2].b = comboMatrix._42 - comboMatrix._22;
		p_planes[2].c = comboMatrix._43 - comboMatrix._23;
		p_planes[2].d = comboMatrix._44 - comboMatrix._24;
		// Bottom clipping plane
		p_planes[3].a = comboMatrix._41 + comboMatrix._21;
		p_planes[3].b = comboMatrix._42 + comboMatrix._22;
		p_planes[3].c = comboMatrix._43 + comboMatrix._23;
		p_planes[3].d = comboMatrix._44 + comboMatrix._24;
		// Near clipping plane
		p_planes[4].a = comboMatrix._41 + comboMatrix._31;
		p_planes[4].b = comboMatrix._42 + comboMatrix._32;
		p_planes[4].c = comboMatrix._43 + comboMatrix._33;
		p_planes[4].d = comboMatrix._44 + comboMatrix._34;
		// Far clipping plane
		p_planes[5].a = comboMatrix._41 - comboMatrix._31;
		p_planes[5].b = comboMatrix._42 - comboMatrix._32;
		p_planes[5].c = comboMatrix._43 - comboMatrix._33;
		p_planes[5].d = comboMatrix._44 - comboMatrix._34;
		// Normalize the plane equations, if requested
		if (normalize == true)
		{
			NormalizePlane(p_planes[0]);
			NormalizePlane(p_planes[1]);
			NormalizePlane(p_planes[2]);
			NormalizePlane(p_planes[3]);
			NormalizePlane(p_planes[4]);
			NormalizePlane(p_planes[5]);
		}
	}
}


