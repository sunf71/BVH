
#include "klbvh.h"
#include "glmModel.h"
#include <vector>
#include <iostream>
#include "timer.h"
#include "frustum.h"
#include "bvh.h"

uint32 loadObj(const char* fileName, std::vector<Primitive>& prims,Bbox3f& BBox)
{
	GLMmodel* model = glmReadOBJ(fileName);
	uint32 num = model->numtriangles;
	prims.resize(num);
	

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
		prims[i] = Primitive(Vector3f(tVec3[0]/3.0,tVec3[1]/3.0,tVec3[2]/3.0));
		prims[i].bbox = box;
		BBox.insert(box);
		
	}


	
	delete model;
	return num;
}
void FrustumCulling(pyrfrustum_t& frustum, BVH& bvh, uint32* out)
{
	
	
	vector<Bvh_Node>* nodes = &bvh.nodes;
	vector<Bvh_Node>* leafs = &bvh.leafs;
	const uint32 stack_size  = 64;
	Bvh_Node* stack[stack_size];
	uint32 top = 0;
	stack[top++] = &(*nodes)[0];
	while(top>0)
	{
		Bvh_Node* node = stack[--top];
		Bbox3f box = bvh.node_Boxes[node->id];
		int ret = IntersectFast(frustum,box);
		if (ret == 2)
		{
			//相交
			if (node->l_isleaf)
			{
				int index = bvh.indices[node->getChild(0)];
				if (IntersectFast(frustum,bvh.leaf_Boxes[index]))
				{
					out[index] = 1;					
				}				
			}
			else
					stack[top++] = &(*nodes)[node->getChild(0)];
			if(node->r_isleaf)
			{
				int index = bvh.indices[node->getChild(1)];
				if (IntersectFast(frustum,bvh.leaf_Boxes[index]))
				{
					out[index] = 1;
					
				}				
			}
			else
					stack[top++] = &(*nodes)[node->getChild(1)];
		}
		else if (ret == 1)
		{
			//in
			for(int k= node->leafStart; k<=node->leafEnd;k++)
				out[bvh.indices[k]] = 1;
		}
	}
	
}

void FrustumCulling(pyrfrustum_t& frustum, Bintree_node* bvh, uint32* out)
{
	
	const uint32 stack_size  = 64;
	uint32 stack[stack_size];
	uint32 top = 0;
	stack[top++] = 0;
	while(top>0)
	{
		uint32 idx = stack[--top];
		Bintree_node * node = &bvh[idx];
		int ret = Intersect(frustum,(float*)(&node->minX));
		if (ret == 2)
		{
			//相交
			
			if(bvh[node->RChild].isLeaf())
			{
				if (Intersect(frustum,(float*)(&bvh[node->RChild].minX)))
				{
					out[bvh[node->RChild].pid] = 1;					
				}				
			}
			else
				stack[top++] = node->RChild;

			if (bvh[node->lChild].isLeaf())
			{
				
				if (Intersect(frustum,(float*)(&bvh[node->lChild].minX)))
				{
					out[bvh[node->lChild].pid] = 1;					
				}				
			}
			else
				stack[top++] = node->lChild;
		}
		else if (ret == 1)
		{
			//in
			for(int k= node->leafStart; k<=node->leafEnd;k++)
				out[k] = 1;
		}
	}
	
}




void CpuKlbvhTest(const thrust::host_vector<Vector3f>& h_points,
	const thrust::host_vector<Bbox3f>& h_boxes, const Bbox3f& gbox,
	pyrfrustum_t& frustum, BVH& bvh)
{
	Bvh_Builder builder;
	Bbox<Vector3f> bbox(Vector3f(0.f),Vector3f(1.f));
	std::vector<Primitive> Primitives;
	for(int i=0; i<h_points.size(); i++)
	{
		Primitive p((Vector3f)h_points[i],(Bbox3f)h_boxes[i]);
		Primitives.push_back(p);
	}
	
	int n = h_points.size();
	/*n = 180;
	for(int i=0; i<n; i++)
	{
		Vector3f p(1.0*(rand()%100)/100,1.0*(rand()%100)/100,1.0*(rand()%100)/100);
		std::cout<<p[0]<<","<<p[1]<<","<<p[2]<<std::endl;
		Primitives.push_back(Primitive(p));
		bbox.insert(p);
	}*/
	//std::vector<Vector3f> h_points;
	//std::vector<Bbox3f> h_boxes;
	//n = loadObj(objFileName,Primitives,bbox);
	Timer timer;
	timer.start();
	builder.build(gbox,Primitives.begin(),Primitives.end(),bvh);
	timer.stop();
	std::cout<<"build time "<<timer.seconds()*1000<<"ms"<<std::endl;
	//bvh.print();
	//bvh.traversal(&printNode);
	std::cout<<"leaf visited "<<globalCounter<<std::endl;

	
	std::vector<uint32> out(bvh.leafs.size(),0);
	timer.start();
	FrustumCulling(frustum,bvh,&out[0]);
	timer.stop();
	std::cout<<"bvh culling "<<timer.seconds()*1000<<"ms"<<std::endl;
	int in = 0;
	for(int i=0; i<out.size(); i++)
		if (out[i] == 1)
			in++;
	std::cout<<"total "<<out.size()<<" in "<<in<<std::endl;

	//brute force
	int k=0;
	std::vector<uint32> out2(bvh.leafs.size(),0);
	timer.start();
	for(int i=0; i<bvh.leaf_Boxes.size(); i++)
		if (Intersect(frustum,bvh.leaf_Boxes[i]))
			out2[i]=1;
	timer.stop();
	in=0;
	for(int i=0; i<out.size(); i++)
		if (out2[i] == 1)
			in++;
	std::cout<<"brute force culling "<<timer.seconds()*1000<<"ms"<<std::endl;
	std::cout<<"brute force in "<<in<<std::endl;
	std::vector<uint32> out3(bvh.leafs.size(),0);
	timer.start();
	for(int i=0; i<bvh.leaf_Boxes.size(); i++)
		if (IntersectFast(frustum,bvh.leaf_Boxes[i]))
			out3[i]=1;
	timer.stop();
	in=0;
	for(int i=0; i<out.size(); i++)
		if (out3[i] == 1)
			in++;
	std::cout<<"fast brute force in "<<in<<std::endl;
	std::cout<<"fast brute force culling "<<timer.seconds()*1000<<"ms"<<std::endl;

	uint32 size = bvh.nodes.size()+bvh.leafs.size();	
	Bintree_node* nbvh = new Bintree_node[size];
	timer.start();
	DFSBinTree(bvh,nbvh);
	timer.stop();
	std::cout<<"nbvh building "<<timer.seconds()*1000<<"ms"<<std::endl;
	
	out = std::vector<uint32>(bvh.leafs.size(),0);
	timer.start();
	FrustumCulling(frustum,nbvh,&out[0]);
	timer.stop();
	std::cout<<"nbvh culling "<<timer.seconds()*1000<<"ms"<<std::endl;
	in = 0;
	for(int i=0; i<out.size(); i++)
		if (out[i] == 1)
			in++;
	std::cout<<"total "<<out.size()<<" in "<<in<<std::endl;
	delete[] nbvh;

}
//int main()
//{
//	
//
//	
//	CpuKlbvhTest();
//
//	return 0;
//}



