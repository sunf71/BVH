#pragma once
#include "basic\types.h"
#include "frustum\frustum.h"

namespace nih
{
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
	void frustumCulling_test(Matrix4x4& mat);
}