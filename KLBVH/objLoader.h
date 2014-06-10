#pragma once
#include "types.h"
#include <thrust\device_vector.h>
#include <thrust\host_vector.h>

#include "bbox.h"
#include "glmModel.h"
//load obj model
//@points [out] 模型每个三角形面片中心
//@boxes [out] 每个三角形的包围盒
uint32 loadObj(const char* fileName, thrust::host_vector<Vector3f>& h_points, thrust::host_vector<Bbox3f>& h_boxes, Bbox3f& BBox)
{
	GLMmodel* model = glmReadOBJ(fileName);
	uint32 num = model->numtriangles;
	h_points.resize(num);
	h_boxes.resize(num);

	for(int i=0; i<num; i++)
	{
		Vector3f p[3];
		Bbox3f box;
		Vector3f tVec3(0,0,0);
		for (int j=0; j<3; j++)
		{
			float* tmp = model->vertices+model->triangles[i].vindices[j]*3;
			tVec3[0]+=*tmp;
			tVec3[1]+=*(tmp+1);
			tVec3[2]+=*(tmp+2);

			p[j] = Vector3f(tmp);
			box.insert(p[j]);
		}
		h_points[i] = Vector3f(tVec3[0]/3.0,tVec3[1]/3.0,tVec3[2]/3.0);
		h_boxes[i] = box;
		BBox.insert(box);		
	}
	delete model;
	return num;
}


void loadRandom(int num, thrust::host_vector<Vector3f>& h_points, thrust::host_vector<Bbox3f>& h_boxes, Bbox3f& BBox)
{
	const float dim = 15.67;
	h_points.resize(num);
	h_boxes.resize(num);
	for(int i=0; i<num; i++)
	{
		Vector3f p[3];
		Bbox3f box;
		Vector3f tVec3(0,0,0);
		for (int j=0; j<3; j++)
		{
			float tmp[] = {rand()%100/100.0*dim,rand()%100/100.0*dim,rand()%100/100.0*dim};
			tVec3[0]+=tmp[0];
			tVec3[1]+=tmp[1];
			tVec3[2]+=tmp[2];

			p[j] = Vector3f(tmp);
			box.insert(p[j]);
		}
		h_points[i] = Vector3f(tVec3[0]/3.0,tVec3[1]/3.0,tVec3[2]/3.0);
		h_boxes[i] = box;
		BBox.insert(box);		
	}

}