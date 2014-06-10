#include <iostream>
#include "bvh\cuda\lbvh_test.h"
#include "frustumCulling_test.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace std;
using namespace nih;
int main()
{	
	cout<<"hello bvh"<<endl;
	
	// traverse both trees top-down to see whether there's any inconsistencies...
	pyrfrustum_t frustum;
	// Projection matrix : 45�� Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
	glm::mat4 Projection = glm::perspective(45.0f, 4.0f / 3.0f, 0.01f, 50.0f);
	// Camera matrix
	glm::mat4 View       = glm::lookAt(
		glm::vec3(0,0,-7), // Camera is at (4,3,3), in World Space
		glm::vec3(0,0,0), // and looks at the origin
		glm::vec3(0,1,0)  // Head is up (set to 0,-1,0 to look upside-down)
		);
	// Model matrix : an identity matrix (model will be at the origin)
	glm::mat4 Model      = glm::mat4(1.0f);  // Changes for each model !
	
	// Our ModelViewProjection : multiplication of our 3 matrices
	glm::mat4 MVP        = Projection * View * Model; // Remember, matrix multiplication is the other way around
	
	Matrix4x4 mvp;		
	
	memcpy(&mvp,&MVP[0][0],16*sizeof(float));
	

	//Intersect(frustum,Bbox4f());
	//lbvh_test();
	frustumCulling_test(mvp);
}
