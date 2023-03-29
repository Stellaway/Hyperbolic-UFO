//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Ursuleac Zsolt
// Neptun : S8H56Y
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"
#include <iostream>

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao, vbo;	   // virtual world on the GPU

const int nTesselatedVertices = 60;

const float hamiSize = 0.2f;

//dot hyperbolic
float dotH(vec3 u, vec3 v) {
	return u.x * v.x + u.y * v.y + -1 * u.z * v.z;
}

//vector length
float absH(vec3 u) {
	return sqrtf(dotH(u, u));
}

//normalize vector
vec3 normalizeH(vec3 u) {
	return u * (1 / absH(u));
}

//perpendicular vector to u
vec3 perpendVec(vec3 p, vec3 u) {
	vec3 perpend;
	//vec3 p = vec3(0, 0, 1);


	return cross(vec3(u.x, u.y, -u.z), vec3(p.x, p.y, -p.z));

	//z=0
	//u.x * perpend.x + u.y * perpend.y = 0
	//u.x * perpend.x = -u.y * perpend.y
	//perpend.x = 1
	//perpend.y = -u.x / u.y
	/*if (u.y == 0) {
		return normalize(vec3(0, 1, 0));
	}
	else if (u.x == 1) {
		return normalize(vec3(1, 0, 0));
	}
	return normalize(vec3(1, -u.x / u.y, 0));*/
}

//point from point p with velocity v for time t
vec3 pointFromPwithVforT(vec3 p, vec3 v, float t) {
	// szeretem a faszt
	return p*coshf(t) + normalize(v)*sinhf(t);
}

//velocity from point p with velocity v for time t
vec3 velocityFromPwithVforT(vec3 p, vec3 v, float t) {
	return normalize(p * sinhf(t) + normalize(v) * coshf(t));
}

//distance between p and q
float dist(vec3 p, vec3 q) {
	//q = p *coshf(t) + normalizeH(v) * sinhf(t);
	return acoshf((dotH(-1 * q, p)));
}

//direction from p to q
vec3 dirTo(vec3 p, vec3 q) {
	return normalize((q - 1 * p * cosh(dist(p, q)) * 1 / sinhf(dist(q, p))));
}

vec3 rotBy(vec3 v, float phi) {
	return normalize(v*cosf(phi) + perpendVec(vec3(0,0,1), v) * sinf(phi));
}

//approximate vector for vector v
vec3 approVec(vec3 v) {
	float lambda = dotH(v, vec3(0, 0, 1));
	return normalize(v + lambda * vec3(0, 0, 1));
}

//approximate point for point p
vec3 approPoint(vec3 p) {
	if (dotH(p,p) < -1.0f) {
		std::cout<<"point: x: "<<p.x<<", y: "<<p.y<<", z: "<<p.z<<std::endl;
		vec3 appro = p * sqrtf(-1.0f / dotH(p, p));
		std::cout << "appro: x: "<<appro.x << ", y: " << appro.y<<", z: "<<appro.z<<std::endl;
		std::cout << "doth(appro, appro)= " << dotH(appro, appro)<<std::endl;
		return p * sqrtf(-1.0f / dotH(p,p));
	}
	if (dotH(p, p) > -1.0f) {
		std::cout << "Nagy a baj, Houston.";
		}
	return p;
}

vec2 projectToPoincareDisk(vec3 v) {
	return vec2(v.x / (v.z + 1), v.y / (v.z + 1));
}


class Circle {
	float radius;
	vec3 center;
	vec3 color;


public:


	void create(float radius, vec3 center, vec3 color) {
		this->radius = radius;
		this->center = vec3(0,0,1);
		this->color = color;
		this->center = approPoint(center);

	}

	void draw() {
		
		std::vector<vec2> projectedPoints;
		
		
		vec2 projected = projectToPoincareDisk(center);

		//v perpendicular for p
		vec3 v = perpendVec(center, vec3(0,1,0));
		
		for (int i = 0; i < nTesselatedVertices; i++) {


			float phi = i * 2.0f * M_PI / nTesselatedVertices;

			

			vec3 circlePoint = pointFromPwithVforT(center, rotBy(v,phi), radius);
			//std::cout<<"circlePoint: "<<circlePoint.x<<" "<<circlePoint.y<<" "<<circlePoint.z<<std::endl;

			projectedPoints.push_back(projectToPoincareDisk(circlePoint));
			
			//write out projected points
			//std::cout << "projected: " << projected.x << " " << projected.y << std::endl;

		}
		/*
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * projectedPoints.size(),  // # bytes
			&projectedPoints[0],	      	// address
			GL_DYNAMIC_DRAW);	// we do not change later

		// Set color to its color
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, color.x, color.y, color.z); // 3 floats
		float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
										  0, 1, 0, 0,    // row-major!
										  0, 0, 1, 0,
										  0, 0, 0, 1 };
		location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Send data to the GPU
		glDrawArrays(GL_TRIANGLE_FAN, 0, projectedPoints.size());*/
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * projectedPoints.size(),  // # bytes
			&projectedPoints[0],	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL);

		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, 0.0f, 0.0f, 1.0f); // 3 floats

		float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
								  0, 1, 0, 0,    // row-major!
								  0, 0, 1, 0,
								  0, 0, 0, 1 };

		location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);

		glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, nTesselatedVertices /*# Elements*/);
	}

};
/*
class Hami {
	vec3 position;
	vec3 direction;
	Circle body;
	Circle wEyes[2];
	Circle bEyes[2];
	Circle mouth;
public:
	void create(vec2 pos, vec2 dir, vec3 color) {
		position = vec3(pos.x, pos.y, 1.0);
		direction = vec3(dir.x, dir.y, 1.0f);
		body.create(hamiSize, vec2(position.x, position.y), color);
		wEyes[0].create(hamiSize / 4, vec2(position.x + hamiSize, position.y + hamiSize / 2), vec3(1, 1, 1));
		wEyes[1].create(hamiSize / 4, vec2(position.x - hamiSize / 2, position.y + hamiSize / 2), vec3(1, 1, 1));

		//bEye.create(hamiSize / 8, vec2(position.x + hamiSize / 4, position.y + hamiSize / 4), vec3(0, 0, 0));
	}
};
*/



std::vector<vec2> PoinCirclePoints;
Circle redCirc;

void drawPoinDisk() {
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(vec2) * PoinCirclePoints.size(),  // # bytes
		&PoinCirclePoints[0],	      	// address
		GL_STATIC_DRAW);	// we do not change later

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
		2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		0, NULL);

	int location = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform3f(location, 0.0f, 0.0f, 0.0f); // 3 floats

	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
							  0, 1, 0, 0,    // row-major!
							  0, 0, 1, 0,
							  0, 0, 0, 1 };

	location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);

	glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, nTesselatedVertices /*# Elements*/);
}

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	glGenVertexArrays(1, &vao);	// get 1 vao id
	glBindVertexArray(vao);		// make it active

	
	glGenBuffers(1, &vbo);	// Generate 1 buffer
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	
	for (int i = 0; i < nTesselatedVertices; i++) {
		float phi = i * 2.0f * M_PI / nTesselatedVertices;
		PoinCirclePoints.push_back(vec2(cosf(phi), sinf(phi)));
	}
	/*
	// Copy to GPU and draw
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
				sizeof(vec2) * PoinCirclePoints.size(),  // # bytes
				&PoinCirclePoints[0],	      	// address
				GL_STATIC_DRAW);	// we do not change later
	
	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
				2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
				0, NULL); 		     // stride, offset: tightly packed
	//glDrawArrays(GL_TRIANGLE_FAN, 0, PoinCirclePoints.size());*/
	
	
	redCirc.create(0.9f, vec3(0,500,501), vec3(1,0,0));
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}





// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.5, 1);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer
	// Set color to (0, 1, 0) = green
	/*int location = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform3f(location, 0.0f, 1.0f, 0.0f); // 3 floats

	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
							  0, 1, 0, 0,    // row-major!
							  0, 0, 1, 0,
							  0, 0, 0, 1 };

	location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);*/	// Load a 4x4 row-major float matrix to the specified location

	//glBindVertexArray(vao);  // Draw call
	//glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, nTesselatedVertices /*# Elements*/);
	// 
	drawPoinDisk();
	redCirc.draw();
	glutSwapBuffers(); // exchange buffers for double buffering
}








// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}







// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}







// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}







// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}







// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}

