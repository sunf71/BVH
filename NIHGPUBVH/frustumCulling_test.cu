
#include "frustumCulling_test.h"
#include "bvh/cuda/lbvh_builder.h"
#include "bintree/bintree_gen.h"
#include "sampling/random.h"
#include "time/timer.h"
#include "basic/cuda_domains.h"
#include "tree/model.h"
#include "tree/cuda/reduce.h"
#include "glmModel.h"
#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>

namespace nih
{
	__global__ void frustumCullingKernel(pyrfrustum_t* frustum, uint32 frustumCount, 
		Bvh_node* d_nodes, uint2* leaves, Bbox4f* nodeBboxes, Bbox4f* leafBboxes, Vector4f* points, char* out)
	{
		//Frustum array index
		int idt = blockDim.x * blockIdx.x + threadIdx.x;
		//

		if( idt > frustumCount )
			return;

		uint32 d_node_id = 0;
		int tmp = 0;
		while (d_node_id != uint32(-1))
		{
			Bvh_node d_node = d_nodes[ d_node_id ];
			if (d_node.is_leaf())
			{

				const uint2 d_leaf = leaves[ d_node.get_leaf_index() ];   

				if (Intersect(frustum[idt],leafBboxes[d_node.get_leaf_index()]))
					for(size_t i=d_leaf.x; i<d_leaf.y;i++)
					{					
						if (FrustumContainPoints( frustum[idt], Vector3f(points[i].x)))
							out[i] = 1;

					}
					d_node_id = d_node.get_skip_node();
			}
			else
			{ 
				if (Intersect(frustum[idt],nodeBboxes[d_node_id]) )					
					d_node_id = d_node.get_child(0);
				else
					d_node_id = d_node.get_skip_node();
			}            
		}
	}
	__global__ void frustumCullingKernel(pyrfrustum_t* frustum, uint32 frustumCount, 
		Bvh_node* d_nodes, uint2* leaves, Bbox4f* nodeBboxes, Bbox4f* leafBboxes, Bbox4f* pBoxes, const uint32 n_p,char* out)
	{
		const size_t maxDepth = 64;
		//Frustum array index
		int idt = blockDim.x * blockIdx.x + threadIdx.x;
		//

		if( idt >= frustumCount )
			return;
		uint32 stack[maxDepth];
		int stack_top = -1;
		stack[++stack_top]  = 0;
		
		while (true)
		{
			if( stack_top < 0 || stack_top == maxDepth )
				break;
			uint32 d_node_id  = stack[ stack_top-- ];
			Bvh_node d_node = d_nodes[d_node_id];
			if (d_node.is_leaf())
			{			

				/*if (Intersect(frustum[idt],leafBboxes[d_node.get_leaf_index()]))
				{*/
					const uint2 d_leaf = leaves[ d_node.get_leaf_index() ];   
					for(size_t i=d_leaf.x; i<d_leaf.y;i++)
					{					
						if (Intersect( frustum[idt], pBoxes[i]))
							out[i+idt*n_p] = 1;
					}		
				/*}*/
			}
			else
			{ 
				if (Intersect(frustum[idt],nodeBboxes[d_node_id]) )	
				{
					for(int i=0; i<d_node.get_child_count(); i++)
						stack[++stack_top] = d_node.get_child(i);
				}		
			}            
		}
	}
	__global__ void BruteforceFrustumCulling(pyrfrustum_t* f, Bbox4f* aabbs, unsigned int primCount, char* out)
{
	//Frustum array index
	int idt =blockIdx.x;
	//

	if( idt > primCount )
		return;

	if( Intersect( *f, aabbs[idt] ) )
	{
		out[ idt ] = 1;
	}
}
	 void frustumCulling(pyrfrustum_t* frustum, uint32 frustumCount, 
		Bvh_node* d_nodes, uint2* leaves, Bbox4f* nodeBboxes, Bbox4f* leafBboxes, Bbox4f* pBoxes, char* out)
	{
		const size_t maxDepth = 64;
		//Frustum array index
		int c = 0;
		uint32 stack[maxDepth];
		int stack_top = -1;
		stack[++stack_top]  = 0;
		
		while (true)
		{
			if( stack_top < 0 || stack_top == maxDepth )
				break;
			uint32 d_node_id  = stack[ stack_top-- ];
			Bvh_node d_node = d_nodes[d_node_id];
			if (d_node.is_leaf())
			{			
				
				/*if (Intersect(frustum[0],leafBboxes[d_node.get_leaf_index()]))
				{*/
					const uint2 d_leaf = leaves[ d_node.get_leaf_index() ];   
					for(size_t i=d_leaf.x; i<d_leaf.y;i++)
					{			
						if (Intersect( frustum[0], pBoxes[i]))
							out[i] = 1;
						
							c++;
					}		
				//}				
			}
			else
			{ 
				c++;
				if (Intersect(frustum[0],nodeBboxes[d_node_id]) )	
				{
					for(int i=0; i<d_node.get_child_count(); i++)
						stack[++stack_top] = d_node.get_child(i);
				}		
			}            
		}
		std::cout<<"culled boxes : "<<c<<std::endl;
	}
	


	 //load obj model
	 //@points [out] 模型每个三角形面片中心
	 //@boxes [out] 每个三角形的包围盒
	 uint32 loadObj(const char* fileName, thrust::device_vector<Vector4f>& points, thrust::device_vector<Bbox4f>& boxes, Bbox3f& BBox)
	 {
		 GLMmodel* model = glmReadOBJ(fileName);
		 uint32 num = model->numtriangles;
		 thrust::host_vector<Vector4f> h_points(num);
		 thrust::host_vector<Bbox4f> h_boxes(num);

		 for(int i=0; i<num; i++)
		 {
			 Vector4f p[3];
			 Bbox4f box;
			 Vector3f tVec3(0,0,0);
			 for (int j=0; j<3; j++)
			 {
				 float* tmp = model->vertices+model->triangles[i].vindices[j]*3;
				 tVec3[0]+=*tmp;
				 tVec3[1]+=*(tmp+1);
				 tVec3[2]+=*(tmp+2);

				 p[j] = Vector4f(*tmp,*(tmp+1),*(tmp+2), 1.f);
				 box.insert(p[j]);
			 }
			 h_points[i] = Vector4f(tVec3[0]/3.0,tVec3[1]/3.0,tVec3[2]/3.0,1.0);
			 h_boxes[i] = box;
			 BBox.insert(Vector3f(box.m_min[0],box.m_min[1],box.m_min[2]));
			 BBox.insert(Vector3f(box.m_max[0],box.m_max[1],box.m_max[2]));
		 }

		
		 points = h_points;
		 boxes = h_boxes;
		 delete model;
		 return num;
	 }
	void printVector(const Vector4f& v)
	{
		std::cout<<v[0]<<","<<v[1]<<","<<v[2]<<std::endl;
	}
	 void loadRandom(uint32 n_points, thrust::device_vector<Vector4f>& d_points, thrust::device_vector<Bbox4f>& d_boxes, Bbox3f& BBox)
	 {
		thrust::host_vector<Vector4f> h_points( n_points );
		thrust::host_vector<Bbox4f> pol(n_points);

		Random random;
		for (uint32 i = 0; i < n_points; ++i)
		{
			h_points[i] = Vector4f( random.next(), random.next(), random.next(), 1.0f );
			/*float offset = random.next();
			pol[i].m_min = h_points[i] - Vector4f(offset,offset,offset,0);
			pol[i].m_max = h_points[i] + Vector4f(offset,offset,offset,0);
			BBox.insert(Bbox3f(Vector3f(&pol[i].m_min[0]),Vector3f(&pol[i].m_max[0])));*/
			/*pol[i].m_max = Vector4f(1.0,1.0,1.0,1.0);
			pol[i].m_min = Vector4f(0.0,0.0,0.0,1.0);*/
		}

		Vector3f min(9999,-5,9999),max(-9999,5,-9999);
		BBox.insert(min);
		BBox.insert(max);
		
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

			//printVector(pol[i].m_min);
			//printVector(pol[i].m_max);
		}
		d_points =  h_points;
		d_boxes = pol;
	 }
	 
	 void frustumCulling_test(Matrix4x4& mat)
	{
		using namespace cub;
		pyrfrustum_t frustum;
		ExtractPlanesGL(frustum.planes,mat,true);
		fprintf(stderr, "lbvh build... started\n");

		
		const uint32 n_tests = 3;
				
		thrust::device_vector<Vector4f> d_points;
		thrust::device_vector<Bbox4f> d_Boxes;
		Bbox3f Bbox;

		//const uint32 n_points = 12486;
		//loadRandom(n_points,d_points,d_Boxes,Bbox);
		uint32 n_points = loadObj("sponza.obj",d_points,d_Boxes,Bbox);
		fprintf(stderr,"number of points %d\n",n_points);

		thrust::device_vector<Vector4f> d_unsorted_points( d_points );

		thrust::device_vector<Bvh_node> bvh_nodes;
		thrust::device_vector<uint2>    bvh_leaves;
		thrust::device_vector<uint32>   bvh_index;

		cuda::LBVH_builder<uint32> builder( bvh_nodes, bvh_leaves, bvh_index );
		
		// Allocate device arrays
		cub::DoubleBuffer<uint32> d_keys;
		cub::DoubleBuffer<Bbox4f> d_values;
		CachingDeviceAllocator  g_allocator(true); 
		//CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[0], sizeof(uint32) * n_points));
		CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(uint32) * n_points));
		d_values.d_buffers[0] = thrust::raw_pointer_cast(&d_Boxes.front());
		CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[1], sizeof(Bbox4f) * n_points));
		// Allocate temporary storage
		size_t  temp_storage_bytes  = 0;
		void    *d_temp_storage     = NULL;
		CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, n_points));
		CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));


		cudaEvent_t start, stop;
		cudaEventCreate( &start );
		cudaEventCreate( &stop );

		float time = 0.0f;

		for (uint32 i = 0; i <= n_tests; ++i)
		{
			d_points = d_unsorted_points;
			cudaThreadSynchronize();

			float dtime;
			cudaEventRecord( start, 0 );

		 /* builder.build(
            Bbox,
            d_points.begin(),
            d_points.end(),
            16u );*/
			//builder.build(
			//	Bbox,
			//	d_points.begin(),
			//	d_points.end(),
			//	d_Boxes.begin(),
			//	d_Boxes.end(),
			//	4u );
			builder.build(
				Bbox,
				d_points.begin(),
				d_points.end(),
				d_Boxes.begin(),
				d_Boxes.end(),
				16u,
				d_keys,
				d_values,
				&d_temp_storage,
				temp_storage_bytes);

			cudaEventRecord( stop, 0 );
			cudaEventSynchronize( stop );
			cudaEventElapsedTime( &dtime, start, stop );

			if (i) // skip the first run
				time += dtime;
		}
		time /= 1000.0f * float(n_tests);

		cudaEventDestroy( start );
		cudaEventDestroy( stop );


		fprintf(stderr, "lbvh test... done\n");
		fprintf(stderr, "  time       : %f ms\n", time * 1000.0f );
		fprintf(stderr, "  points/sec : %f M\n", (n_points / time) / 1.0e6f );

		fprintf(stderr, "  nodes  : %u\n", builder.m_node_count );
		fprintf(stderr, "  leaves : %u\n", builder.m_leaf_count );
		for (uint32 level = 0; level < 30; ++level)
			fprintf(stderr, "  level %u : %u nodes\n", level, builder.m_levels[level+1] - builder.m_levels[level] );

		fprintf(stderr, "lbvh bbox reduction test... started\n");

		BFTree<Bvh_node*,device_domain> bvh(
			thrust::raw_pointer_cast( &bvh_nodes.front() ),
			builder.m_leaf_count,
			thrust::raw_pointer_cast( &bvh_leaves.front() ),
			30u,
			builder.m_levels );

		thrust::device_vector<Bbox4f> d_leaf_bboxes( builder.m_leaf_count );
		thrust::device_vector<Bbox4f> d_node_bboxes( builder.m_node_count );

		cudaEventCreate( &start );
		cudaEventCreate( &stop );

		time = 0.0f;

		for (uint32 i = 0; i <= n_tests; ++i)
		{
			float dtime;
			cudaEventRecord( start, 0 );

			/*cuda::tree_reduce(
				bvh,
				thrust::raw_pointer_cast( &d_points.front() ),
				thrust::raw_pointer_cast( &d_leaf_bboxes.front() ),
				thrust::raw_pointer_cast( &d_node_bboxes.front() ),
				cuda::bbox_functor(),
				Bbox4f() );*/
			/*cuda::tree_reduce(
				bvh,
				thrust::raw_pointer_cast( &d_Boxes.front() ),
				thrust::raw_pointer_cast( &d_leaf_bboxes.front() ),
				thrust::raw_pointer_cast( &d_node_bboxes.front() ),
				cuda::bbox_functor(),
				Bbox4f());*/
			cuda::tree_reduce(
				bvh,
				d_values.d_buffers[1],
				thrust::raw_pointer_cast( &d_leaf_bboxes.front() ),
				thrust::raw_pointer_cast( &d_node_bboxes.front() ),
				cuda::bbox_functor(),
				Bbox4f());

			cudaEventRecord( stop, 0 );
			cudaEventSynchronize( stop );
			cudaEventElapsedTime( &dtime, start, stop );

			if (i) // skip the first run
				time += dtime;
		}
		time /= 1000.0f * float(n_tests);

		cudaEventDestroy( start );
		cudaEventDestroy( stop );

		fprintf(stderr, "lbvh bbox reduction test... done\n");
		fprintf(stderr, "  time       : %f ms\n", time * 1000.0f );
		fprintf(stderr, "  points/sec : %f M\n", (n_points / time) / 1.0e6f );

		thrust::host_vector<Bvh_node> d_nodes( bvh_nodes );
		thrust::host_vector<uint2>    d_leaves( bvh_leaves );
		thrust::host_vector<Bbox4f> h_nodeBoxes(d_node_bboxes);
		thrust::host_vector<Bbox4f> h_leafBoxes(d_leaf_bboxes);
		//thrust::host_vector<Bbox4f> h_PBoxes(d_Boxes);
		thrust::host_vector<Bbox4f> h_PBoxes(d_Boxes.size());
		cudaMemcpy(&h_PBoxes.front(),d_values.d_buffers[1],sizeof(Bbox4f)*d_Boxes.size(),cudaMemcpyDeviceToHost);
		std::cout<<" node boxes size : "<<h_nodeBoxes.size()<<std::endl;
		std::cout<<" leaf boxes size : "<<h_leafBoxes.size()<<std::endl;
	
		//std::cout<<" points boxes"<<std::endl;
		//for(int i=0; i<h_PBoxes.size(); i++)
		//{
		//	printVector(h_PBoxes[i].m_min);
		//	printVector(h_PBoxes[i].m_max);
		//}
		//std::cout<<" node boxes"<<std::endl;
		//for(int i=0; i<h_nodeBoxes.size(); i++)
		//{
		//	printVector(h_nodeBoxes[i].m_min);
		//	printVector(h_nodeBoxes[i].m_max);
		//}
		//std::cout<<" leaf boxes"<<std::endl;
		//for(int i=0; i<h_leafBoxes.size(); i++)
		//{
		//	printVector(h_leafBoxes[i].m_min);
		//	printVector(h_leafBoxes[i].m_max);
		//}
		std::cout<<"culling by device  "<<std::endl;
		uint32 d_node_id = 0;
		//遍历打印bvh
		/*while (d_node_id != uint32(-1))
		{
			Bvh_node d_node = d_nodes[ d_node_id ];
			if (d_node.is_leaf())
			{
				
				const uint2 d_leaf = d_leaves[ d_node.get_leaf_index() ];   
				std::cout<<"叶子节点 "<<d_leaf.x<<"~"<<d_leaf.y<<std::endl;
				
				d_node_id = d_node.get_skip_node();
			}
			else
			{ 
				std::cout<<"中间节点"<<d_node.get_index() <<"包含"<<d_node.get_child_count()<<"个儿子"<<std::endl;
				for(int i=0; i<d_node.get_child_count(); i++)
				{
					if (d_nodes[d_node.get_child(i)].is_leaf())
						std::cout<<"第"<<i<<"个儿子是叶节点"<<std::endl;
					else
						std::cout<<"第"<<i<<"个儿子 "<<d_nodes[d_node.get_child(i)].get_index()<<std::endl;
				}
				
				d_node_id = d_node.get_child(0);
					
			}            
		}*/
		d_node_id = 0;
		thrust::host_vector<uint32> in;
		unsigned int c = 0;
		while (d_node_id != uint32(-1))
		{
			Bvh_node d_node = d_nodes[ d_node_id ];
			if (d_node.is_leaf())
			{
				
				const uint2 d_leaf = d_leaves[ d_node.get_leaf_index() ];  
				//std::cout<<d_node.get_index()<<"是叶子节点 "<<d_leaf.x<<"~"<<d_leaf.y<<std::endl;				
				if (Intersect(frustum,h_leafBoxes[d_node.get_leaf_index()]))
				{
					for(size_t i=d_leaf.x; i<d_leaf.y;i++)
					{					
						/*if (FrustumContainPoints( frustum, Vector3f(h_points[i].x)))*/						
						if (Intersect( frustum, h_PBoxes[i]))
							in.push_back(i);
					}
				}
				else
				{
					c++;
				}
				d_node_id = d_node.get_skip_node();
			}
			else
			{ 
				
				if (Intersect(frustum,h_nodeBoxes[d_node_id]) )					
					d_node_id = d_node.get_child(0);
				else
				{
					d_node_id = d_node.get_skip_node();
					c++;
				}
			}            
		}
		std::cout<<"in "<<in.size()<<std::endl;
		/*for(int i=0; i< in.size(); i++)
			std::cout<<in[i]<<" "<<std::endl;*/
		std::cout<<"intesects "<<c<<std::endl;

		////char out3[n_points] = {0};
		//char * out3 = new char[n_points];
		//frustumCulling(&frustum,1,&d_nodes.front(),&d_leaves.front(),&h_nodeBoxes.front(),&h_leafBoxes.front(),&h_PBoxes.front(),out3);
	
		//视锥体的个数
		uint32 fn = 1;
		thrust::device_vector<char> d_in(n_points*fn,0);
		char * out = thrust::raw_pointer_cast(&d_in.front());
		Bvh_node * Bvh_nodes = thrust::raw_pointer_cast ( &bvh_nodes.front() );
		uint2*    Bvh_leaves = thrust::raw_pointer_cast( &bvh_leaves.front() );
		Vector4f* points = thrust::raw_pointer_cast( &d_points.front() );
		Bbox4f* leafBoxes =  thrust::raw_pointer_cast( &d_leaf_bboxes.front() );
		Bbox4f* nodeBoxes = 		thrust::raw_pointer_cast( &d_node_bboxes.front() );
		//Bbox4f* pBoxes = thrust::raw_pointer_cast(&d_Boxes.front());
		Bbox4f* pBoxes = d_values.d_buffers[1];
		thrust::device_ptr<pyrfrustum_t> d_frustum = thrust::device_malloc<pyrfrustum_t>(1);
		pyrfrustum_t * raw_d_frustum = thrust::raw_pointer_cast(d_frustum);
		cudaMemcpy(raw_d_frustum,&frustum,sizeof(pyrfrustum_t),cudaMemcpyHostToDevice);
		//
		thrust::device_vector<pyrfrustum_t> frustums(fn,frustum);

		cudaEventCreate( &start );
		cudaEventCreate( &stop );
		float dtime;
		cudaEventRecord( start, 0 );
		frustumCullingKernel<<<fn,1>>>(thrust::raw_pointer_cast(&frustums.front()),fn, Bvh_nodes,Bvh_leaves,nodeBoxes,leafBoxes, pBoxes,n_points,out);
		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &dtime, start, stop );
		fprintf(stderr, "culling by kernel time       : %f ms\n", dtime);
		thrust::host_vector<char> h_out(d_in);
		uint32 tmp = 0;
		for(int i=0; i<h_out.size(); i++)
			if( h_out[i] ==1)
				tmp++;
		std::cout<<tmp<<std::endl;

		thrust::device_vector<char> d_in2(n_points,0);
		char * out2 = thrust::raw_pointer_cast(&d_in2.front());
		//cudaEventCreate( &start );
		//cudaEventCreate( &stop );
		//
		cudaEventRecord( start, 0 );
		BruteforceFrustumCulling<<<n_points,1>>>(raw_d_frustum,pBoxes,n_points,out2);
		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &dtime, start, stop );
		fprintf(stderr, "brute force culling  time       : %f ms\n", dtime);
		thrust::host_vector<char> h_out2(d_in2);
		uint32 tmp2 = 0;
		for(int i=0; i<h_out2.size(); i++)
			if( h_out2[i] ==1)
				tmp2++;
		std::cout<<tmp2;

		//thrust::device_free(d_frustum);
		
		//delete[] out3;

		//clean
		//if (d_keys.d_buffers[0]) CubDebugExit(allocator.DeviceFree(d_keys.d_buffers[0]));
		if (d_keys.d_buffers[1]) CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[1]));
		//if (d_values.d_buffers[0]) CubDebugExit(allocator.DeviceFree(d_values.d_buffers[0]));
		if (d_values.d_buffers[1]) CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[1]));
		if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
	}

	
}
