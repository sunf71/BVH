
#include "frustumCulling_test.h"
#include "bvh/cuda/lbvh_builder.h"
#include "bintree/bintree_gen.h"
#include "sampling/random.h"
#include "time/timer.h"
#include "basic/cuda_domains.h"
#include "tree/model.h"
#include "tree/cuda/reduce.h"



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

		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		//

		if( tid > primCount )
			return;

		if( Intersect( *f, aabbs[tid] ) )
		{
			out[ tid ] = 1;
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



	void frustumCulling_test(Matrix4x4& mat)
	{
		pyrfrustum_t frustum;
		ExtractPlanesGL(frustum.planes,mat,true);
		fprintf(stderr, "lbvh build... started\n");

		const uint32 n_points = 12486;
		const uint32 n_tests = 3;

		thrust::host_vector<Vector4f> h_points( n_points );

		Random random;
		for (uint32 i = 0; i < n_points; ++i)
			h_points[i] = Vector4f( random.next(), random.next(), random.next(), 1.0f );
		Vector3f min(9999,-5,9999),max(-9999,5,-9999);
		thrust::host_vector<Bbox4f> pol(n_points);
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
		thrust::device_vector<Vector4f> d_points( h_points );
		thrust::device_vector<Bbox4f> d_Boxes(pol);
		thrust::device_vector<Vector4f> d_unsorted_points( h_points );

		thrust::device_vector<Bvh_node> bvh_nodes;
		thrust::device_vector<uint2>    bvh_leaves;
		thrust::device_vector<uint32>   bvh_index;

		cuda::LBVH_builder<uint32> builder( bvh_nodes, bvh_leaves, bvh_index );

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

			/*builder.build(
			Bbox3f( min, max ),
			d_points.begin(),
			d_points.end(),
			2u );*/
			builder.build(
				Bbox3f( min, max ),
				d_points.begin(),
				d_points.end(),
				d_Boxes.begin(),
				d_Boxes.end(),
				32u );

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
			cuda::tree_reduce(
				bvh,
				thrust::raw_pointer_cast( &d_Boxes.front() ),
				thrust::raw_pointer_cast( &d_leaf_bboxes.front() ),
				thrust::raw_pointer_cast( &d_node_bboxes.front() ),
				cuda::bbox_functor(),
				Bbox4f() );

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
		thrust::host_vector<Bbox4f> h_PBoxes(d_Boxes);
		std::cout<<" node boxes size : "<<h_nodeBoxes.size()<<std::endl;
		std::cout<<" leaf boxes size : "<<h_leafBoxes.size()<<std::endl;

		uint32 d_node_id = 0;
		thrust::host_vector<uint32> in;
		unsigned int c = 0;
		while (d_node_id != uint32(-1))
		{
			Bvh_node d_node = d_nodes[ d_node_id ];
			if (d_node.is_leaf())
			{

				const uint2 d_leaf = d_leaves[ d_node.get_leaf_index() ];   
				c++;
				if (Intersect(frustum,h_leafBoxes[d_node.get_leaf_index()]))
					for(size_t i=d_leaf.x; i<d_leaf.y;i++)
					{					
						/*if (FrustumContainPoints( frustum, Vector3f(h_points[i].x)))*/
						c++;
						if (Intersect( frustum, h_PBoxes[i]))
							in.push_back(i);
					}
					d_node_id = d_node.get_skip_node();
			}
			else
			{ 
				c++;
				if (Intersect(frustum,h_nodeBoxes[d_node_id]) )					
					d_node_id = d_node.get_child(0);
				else
					d_node_id = d_node.get_skip_node();
			}            
		}
		std::cout<<"in "<<in.size()<<std::endl;
		std::cout<<c<<std::endl;
		char out3[n_points] = {0};
		frustumCulling(&frustum,1,&d_nodes.front(),&d_leaves.front(),&h_nodeBoxes.front(),&h_leafBoxes.front(),&h_PBoxes.front(),out3);
		//视锥体的个数
		uint32 fn = 1;
		thrust::device_vector<char> d_in(h_points.size()*fn,0);
		char * out = thrust::raw_pointer_cast(&d_in.front());
		Bvh_node * Bvh_nodes = thrust::raw_pointer_cast ( &bvh_nodes.front() );
		uint2*    Bvh_leaves = thrust::raw_pointer_cast( &bvh_leaves.front() );
		Vector4f* points = thrust::raw_pointer_cast( &d_points.front() );
		Bbox4f* leafBoxes =  thrust::raw_pointer_cast( &d_leaf_bboxes.front() );
		Bbox4f* nodeBoxes = 		thrust::raw_pointer_cast( &d_node_bboxes.front() );
		Bbox4f* pBoxes = thrust::raw_pointer_cast(&d_Boxes.front());
		thrust::device_ptr<pyrfrustum_t> d_frustum = thrust::device_malloc<pyrfrustum_t>(1);
		pyrfrustum_t * raw_d_frustum = thrust::raw_pointer_cast(d_frustum);
		cudaMemcpy(raw_d_frustum,&frustum,sizeof(pyrfrustum_t),cudaMemcpyHostToDevice);

		thrust::device_vector<pyrfrustum_t> frustums(fn,frustum);

		cudaEventCreate( &start );
		cudaEventCreate( &stop );
		float dtime;
		cudaEventRecord( start, 0 );
		frustumCullingKernel<<<fn,1>>>(thrust::raw_pointer_cast(&frustums.front()),fn, Bvh_nodes,Bvh_leaves,nodeBoxes,leafBoxes, pBoxes,n_points,out);
		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &dtime, start, stop );
		fprintf(stderr, "culling  time       : %f ms\n", dtime);
		thrust::host_vector<char> h_out(d_in);
		uint32 tmp = 0;
		for(int i=0; i<h_out.size(); i++)
			if( h_out[i] ==1)
				tmp++;
		std::cout<<tmp<<std::endl;

		thrust::device_vector<char> d_in2(h_points.size(),0);
		char * out2 = thrust::raw_pointer_cast(&d_in2.front());
		cudaEventCreate( &start );
		cudaEventCreate( &stop );

		cudaEventRecord( start, 0 );
		//BruteforceFrustumCulling<<<n_points,1>>>(raw_d_frustum,pBoxes,n_points,out2);
		BFFrustumCulling(raw_d_frustum,pBoxes,n_points,out2);
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

		thrust::device_free(d_frustum);
	}


}


void BFFrustumCulling(nih::pyrfrustum_t* f, nih::Bbox4f* aabbs, unsigned int primCount, char* out)
{
	using namespace nih;
	const uint32 BLOCK_SIZE = 1024;
	const uint32 max_blocks = 65535;
	const size_t n_blocks   = nih::min( max_blocks, (primCount + BLOCK_SIZE-1) / BLOCK_SIZE );
	thrust::device_ptr<pyrfrustum_t> d_frustum = thrust::device_malloc<pyrfrustum_t>(1);
	pyrfrustum_t * raw_d_frustum = thrust::raw_pointer_cast(d_frustum);
	cudaMemcpy(raw_d_frustum,f,sizeof(pyrfrustum_t),cudaMemcpyHostToDevice);

	thrust::device_vector<char> d_out(primCount,0);

	thrust::device_vector<Bbox4f> d_aabbs(primCount);
	Bbox4f* raw_d_aabbs = thrust::raw_pointer_cast(&d_aabbs.front());
	cudaMemcpy(raw_d_aabbs,aabbs,sizeof(Bbox4f)*primCount,cudaMemcpyHostToDevice);

	BruteforceFrustumCulling<<<n_blocks,BLOCK_SIZE>>>(raw_d_frustum,raw_d_aabbs,primCount,thrust::raw_pointer_cast(&d_out.front()));

	thrust::device_free(d_frustum);

	cudaMemcpy(out,thrust::raw_pointer_cast(&d_out.front()),primCount,cudaMemcpyDeviceToHost);
}