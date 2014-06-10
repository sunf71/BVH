
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "frustum.h"
#include "bbox.h"
#include "random.h"
#include <vector>
#include <stdio.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
_declspec(dllimport) cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
_declspec(dllimport) void BFFrustumCulling(nih::pyrfrustum_t* f, nih::Bbox4f* aabbs, unsigned int primCount, char* out);

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
	nih::plane_t * p_planes,
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

void TestCulling()
{
	using namespace nih;
	// Projection matrix : 45бу Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
	glm::mat4 Projection = glm::perspective(45.0f, 4.0f / 3.0f, 0.1f, 100.0f);
	// Camera matrix
	glm::mat4 View       = glm::lookAt(
		glm::vec3(0,0,10), // Camera is at (4,3,3), in World Space
		glm::vec3(0,0,0), // and looks at the origin
		glm::vec3(0,1,0)  // Head is up (set to 0,-1,0 to look upside-down)
		);
	// Model matrix : an identity matrix (model will be at the origin)
	glm::mat4 Model      = glm::mat4(1.0f);  // Changes for each model !

	// Our ModelViewProjection : multiplication of our 3 matrices
	glm::mat4 MVP        = Projection * View * Model; // Remember, matrix multiplication is the other way around

	Matrix4x4 mvp;		

	memcpy(&mvp,&MVP[0][0],16*sizeof(float));

	nih::pyrfrustum_t frustum;
	ExtractPlanesGL(frustum.planes,mvp,true);

	const uint32 n_points = 12486;
	

		//
		std::vector<Vector4f> h_points( n_points );

		nih::Random random;
		for (uint32 i = 0; i < n_points; ++i)
			h_points[i] = Vector4f( random.next(), random.next(), random.next(), 1.0f );
		Vector3f min(9999,-5,9999),max(-9999,5,-9999);
		std::vector<Bbox4f> pol(n_points);
		for( int i = 0; i < n_points; ++i )
		{
			float x = -45 + rand()%100;
			float z = -45 + rand()%100;
			pol[ i ].m_min[0] = x - 5;
			if (pol[ i ].m_min[0] < min[0])
				min[0] = pol[ i ].m_min[0];		
			pol[ i ].m_min[1] = -5;
			pol[ i ].m_min[2] = z - 5;
			if (pol[ i ].m_min[2] < min[2])
				min[2] = pol[ i ].m_min[2];
			pol[ i ].m_max[0] = x + 5;
			if (pol[ i ].m_max[0] >max[0])
				max[0] = pol[ i ].m_max[0];
			pol[ i ].m_max[1] = 5;
			pol[ i ].m_max[2] = z + 5;
			pol[i].m_max[3] = pol[i].m_min[3] = 1.0f;
			if (pol[ i ].m_max[2] >max[2])
				max[2] = pol[ i ].m_max[2];
			for(int j=0; j<3; j++)
				h_points[ i ][j] = (pol[ i ].m_min[j] + pol[ i ].m_max[j])*0.5;
			h_points[i][3] = 1.0f;

		}
		char* out = new char[n_points];
		BFFrustumCulling(&frustum,&pol[0],n_points,out);

		unsigned int c = 0;
		for(int i=0; i<n_points; i++)
			if(out[i] == 1)
				c++;
		std::cout<<c<<std::endl;
		delete[] out;
}
int main()
{
	
	TestCulling();
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}


