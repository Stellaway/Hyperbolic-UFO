//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
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

const int nTesselatedVertices = 80;

const float hamiSize = 0.5f;

void printvector(vec3 p) {
	printf("\nx: %lf, y: %lf, w: %lf\n", p.x, p.y, p.z);
}
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

//approximate point for point p
vec3 approPoint(vec3 p) {
	if (dotH(p, p) < 0.0f) {
		vec3 appro = p * sqrtf(-1.0f / dotH(p, p));

		return p * sqrtf(-1.0f / dotH(p, p));
	}
	if (dotH(p, p) > 0.0f) {
		return p * sqrtf(1.0f / dotH(p, p));
	}
	return p;
}
//projected vector for vector v in point on the plane p
vec3 approVec(vec3 v, vec3 p) {

	if (dotH(p, v) == 0) return v;

	float lambda = dotH(v, p);
	return normalizeH(v + lambda * p);
}

//perpendicular vector to u
vec3 perpendVec(vec3 p, vec3 u) {
	return normalizeH(cross(vec3(u.x, u.y, -u.z), vec3(p.x, p.y, -p.z)));
}

//point from point p with velocity v for time t
vec3 pointFromPwithVforT(vec3 p, vec3 v, float t) {
	vec3 newP = p*coshf(t) + normalizeH(v)*sinhf(t);
	return approPoint(newP);
}

//velocity from point p with velocity v for time t
vec3 velocityFromPwithVforT(vec3 p, vec3 v, float t) {
	vec3 newV = p * sinhf(t) + normalizeH(v) * coshf(t);
	vec3 newP = pointFromPwithVforT(p, v, t);

	newV = approVec(newV, newP);

	return normalizeH(newV);
}

//distance between p and q
float dist(vec3 p, vec3 q) {
	return acoshf(-1*dotH( q, p));
}

//direction from p to q
vec3 dirTo(vec3 p, vec3 q) {
	return normalizeH((q - 1 * p * cosh(dist(p, q))) * 1 / sinhf(dist(q, p)));
}

vec3 rotBy(vec3 p, vec3 v, float phi) {
	v = normalizeH(v);
	return normalizeH(v*cosf(phi) + perpendVec(p, v) * sinf(phi));
}

vec2 projectToPoincareDisk(vec3 v) {
	return vec2(v.x / (v.z + 1), v.y / (v.z + 1));
}

void drawLineStrip(std::vector<vec2> points) {
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(vec2) * points.size(),  // # bytes
		&points[0],	      	// address
		GL_STATIC_DRAW);	// we do not change later

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
		2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		0, NULL);

	int location = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform3f(location, 1.0f, 1.0f, 1.0f); // 3 floats

	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
							  0, 1, 0, 0,    // row-major!
							  0, 0, 1, 0,
							  0, 0, 0, 1 };

	location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);

	glDrawArrays(GL_LINE_STRIP, 0 /*startIdx*/, points.size() /*# Elements*/);
}

class Circle {
public:
	float initialRadius;
	float radius;
	vec3 center;
	vec3 color;

	void create(float radius, vec3 center, vec3 color) {
		this->radius = radius;
		this->center = approPoint(center);
		this->color = color;
		this->initialRadius = radius;

	}

	vec3 update(vec3 v, float t) {
		vec3 oldCenter = center;
		center = pointFromPwithVforT(center, v, t);
		return velocityFromPwithVforT(oldCenter, v, t);
	}

	void setRadius(float rad) {
		radius = rad;
	}

	void draw() {
		
		std::vector<vec2> projectedPoints;
		
		
		vec2 projected = projectToPoincareDisk(center);

		//v perpendicular for p
		vec3 v = perpendVec(center, vec3(0,1,0));
		
		for (int i = 0; i < nTesselatedVertices; i++) {

			float phi = i * 2.0f * M_PI / nTesselatedVertices;

			vec3 circlePoint = pointFromPwithVforT(center, rotBy(center, v,phi), radius);
			
			projectedPoints.push_back(projectToPoincareDisk(circlePoint));
			
		}
		
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
					sizeof(vec2) * projectedPoints.size(),  // # bytes
					&projectedPoints[0],	      	// address
					GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(	0,       // vbo -> AttribArray 0
								2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
								0, NULL);

		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, color.x,color.y, color.z); // 3 floats

		float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
								  0, 1, 0, 0,    // row-major!
								  0, 0, 1, 0,
								  0, 0, 0, 1 };

		location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);

		glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, nTesselatedVertices /*# Elements*/);
	}

};

class Hami {
public:
	vec3 position;
	vec3 direction;
	Circle body;
	Circle wEyes[2];
	Circle bEyes[2];
	Circle mouth;
	std::vector<vec2> hamiPoints;
	void set(vec3 pos, vec3 dir, vec3 color) {
		position = approPoint(pos);
		direction = approVec(dir, position);
		body.create(hamiSize, pos, color);wEyes[0].create(hamiSize / 4.0f, pointFromPwithVforT(position, rotBy(position, direction, 35.0f / 180.0f * M_PI), hamiSize), vec3(1, 1, 1));
		wEyes[1].create(hamiSize / 4.0f, pointFromPwithVforT(position, rotBy(position, direction, -35.0f / 180.0f * M_PI), hamiSize), vec3(1, 1, 1));
		mouth.create(hamiSize / 3.0f, pointFromPwithVforT(position, rotBy(position, direction, 0.0f), hamiSize), vec3(0, 0, 0));
	}
	void drawHami() {
		body.draw();
		mouth.draw();
		wEyes[0].draw();
		wEyes[1].draw();
		bEyes[0].draw();
		bEyes[1].draw();
	}

	void setMouth(float lambda) {
		mouth.setRadius(mouth.initialRadius * lambda);
	}

	void drawHamiPoints() {
		
		drawLineStrip(hamiPoints);
	}

	void alertHamiEye(vec3 p){
		bEyes[0].create(wEyes[0].radius / 4.0f, pointFromPwithVforT(wEyes[0].center, dirTo(wEyes[0].center, p), hamiSize / 5.2f), vec3(0, 0.7f, 1));
		bEyes[1].create(wEyes[1].radius / 4.0f, pointFromPwithVforT(wEyes[1].center, dirTo(wEyes[1].center, p), hamiSize / 5.2f), vec3(0, 0.7f, 1));

	}

	void goForT(float t) {
		hamiPoints.push_back(projectToPoincareDisk(body.center));
		direction = body.update(direction, t);
		set(body.center, direction, body.color);
	}

	void changeDir(float phi) {
		direction = rotBy(position, direction, phi);
		set(position, direction, body.color);
	}

};

void matchEyes(Hami h1, Hami h2) {

}

std::vector<vec2> PoinCirclePoints;
Hami redHami;
Hami greenHami;

void drawVector(vec3 p, vec3 v) {
	std::vector<vec2> vectorPoints;
	vec3 newP = pointFromPwithVforT(p, v, 0.5f);
	vectorPoints.push_back(projectToPoincareDisk(p));
	vectorPoints.push_back(projectToPoincareDisk(newP));
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(vec2) * vectorPoints.size(),  // # bytes
		&vectorPoints[0],	      	// address
		GL_STATIC_DRAW);	// we do not change later

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
		2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		0, NULL);

	int location = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform3f(location, 0.0f, 1.0f, 0.0f); // 3 floats

	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
							  0, 1, 0, 0,    // row-major!
							  0, 0, 1, 0,
							  0, 0, 0, 1 };

	location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);

	glDrawArrays(GL_LINE_STRIP, 0 /*startIdx*/, 2 /*# Elements*/);
}

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
	glLineWidth(1.5f);

	glGenVertexArrays(1, &vao);	// get 1 vao id
	glBindVertexArray(vao);		// make it active

	
	glGenBuffers(1, &vbo);	// Generate 1 buffer
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	
	for (int i = 0; i < nTesselatedVertices; i++) {
		float phi = i * 2.0f * M_PI / nTesselatedVertices;
		PoinCirclePoints.push_back(vec2(cosf(phi), sinf(phi)));
	}
	
	redHami.set(vec3(0,0, 1), vec3(0, 1,0), vec3(1, 0, 0));
	greenHami.set(vec3(3, 0, 3.16f), vec3(-1, 1, 0), vec3(0, 1, 0));
	redHami.alertHamiEye(greenHami.position);
	greenHami.alertHamiEye(redHami.position);

	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.5, 1);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer
	
	drawPoinDisk();
	greenHami.drawHamiPoints();
	redHami.drawHamiPoints();
	greenHami.drawHami();
	redHami.drawHami();
	
	glutSwapBuffers(); // exchange buffers for double buffering
}

bool pressed[256] = { false, };

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	pressed[key] = true;
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
	pressed[key] = false;
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
}

long lastTime = 0;
long deltaT = 0;

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	int defDelta = 30; // h�ny miliszekundumonk�nt legyen k�pfriss�t�s (30 = 33.3 fps, 17 = 60 fps)

	deltaT += time - lastTime;
	for (deltaT; deltaT >= defDelta; deltaT -= defDelta) {
		if (pressed['e']) { redHami.goForT(.100f);		redHami.alertHamiEye(greenHami.position); }
		if (pressed['s']) { redHami.changeDir(.15f);	redHami.alertHamiEye(greenHami.position); }
		if (pressed['f']) { redHami.changeDir(-.15f);	redHami.alertHamiEye(greenHami.position); }

		redHami.setMouth(abs(sinf(.5f * time / 100.0f)));
		greenHami.setMouth(abs(sinf(.5f * time / 100.0f)));

		greenHami.goForT(.100f);
		greenHami.changeDir(.15f);

		redHami.alertHamiEye(greenHami.position);
		greenHami.alertHamiEye(redHami.position);

		greenHami.setMouth(abs(sinf(.5f * time / 100.0f)));

	}

	lastTime = time;
	glutPostRedisplay(); // redraw the scene
	
}

