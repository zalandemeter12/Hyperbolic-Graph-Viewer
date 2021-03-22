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
// Nev    : Demeter Zal�n
// Neptun : VERF1U
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

#define FLT_MIN				1.175494351e-38F        // min normalized positive value (used for coordinate corrections)
#define NUM_OF_VERTICES		50						// number of vertices
#define NUM_OF_CONNECTIONS	61						// number of connections

// vertex shader in GLSL
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;								// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec3 vp;				// Varying input: vp = vertex position is expected in attrib array 0
	layout(location = 1) in vec2 vertexUV;			// Attrib Array 1

	out vec2 texCoord;								// output attribute

	void main() {
		texCoord = vertexUV;
		gl_Position = vec4(vp.x/vp.z, vp.y/vp.z, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
	precision highp float;

	uniform int isTextured;
	uniform sampler2D texturer;
	uniform vec3 color;		// uniform variable, the color of the primitive
	in vec2 texCoord;			// variable input: interpolated texture coordinates
	out vec4 outColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		if (isTextured != 0) {
			outColor = texture(texturer, texCoord);
		} else {
			outColor = vec4(color, 1);	// computed color is the color of the primitive
		}
	}
)";

GPUProgram gpuProgram;	// vertex and fragment shaders

class Graph {	
private:
	struct Pair {
		unsigned int x, y;
		Pair(unsigned int x0 = 0, unsigned int y0 = 0) { x = x0; y = y0; }
		inline bool operator==(const Pair& p) {
			if ((x == p.x && y == p.y) || (x == p.y && y == p.x)) return true;
			else return false;
		}
	};
	unsigned int vao0, vao1;
	unsigned int vbo0;
	unsigned int vbo1[2];
	Pair connections[NUM_OF_CONNECTIONS];
	vec3 vertices[NUM_OF_VERTICES];
	vec3 velocities[NUM_OF_VERTICES];
	vec3 connected[NUM_OF_CONNECTIONS * 2];
	float colors[NUM_OF_VERTICES][6];
public:
	bool startClustering = false;
	bool startDynamicSimulation = false;
	float simulationCycle = 0;
	int drawCycle = 0;
	int drawCount = 2;
	float T = 8;
	float dt = 0.05;
	int rho = 2;
	int m = 1;
	float damping = 0.95;

	Graph() {
		srand(1257789818522);
		vao0 = NULL; vao1 = NULL;
		for (int i = 0; i < NUM_OF_VERTICES; ++i) {
			float x = randomNumber(1,-1);
			float y = randomNumber(1, -1);
			vertices[i] = vec3(x, y, sqrtf(x * x + y * y + 1));
		}
		int i = 0;
		while (i < NUM_OF_CONNECTIONS) {
			int idx1 = rand() % NUM_OF_VERTICES, idx2 = rand() % NUM_OF_VERTICES;
			if (idx1 == idx2) continue;
			if (inConnections(Pair(idx1, idx2))) continue;
			connections[i] = Pair(idx1, idx2);
			++i;
		}

		int j = 0;
		while (j <NUM_OF_VERTICES) {
			colors[j][0] = randomNumber(1, 0);
			colors[j][1] = randomNumber(1, 0);
			colors[j][2] = randomNumber(1, 0);
			colors[j][3] = randomNumber(1, 0);
			colors[j][4] = randomNumber(1, 0);
			colors[j][5] = randomNumber(1, 0);
			bool tooClose = false;
			for (int k = 0; k < j; k += 6) {
				if (colorDistance(k, j) < 1.2) { tooClose = true; break; }
			}
			if (tooClose) continue;
			
			++j;
		}
		
	}

	float colorDistance(int i, int j) {
		float tmp1 = pow(colors[i][0] - colors[j][0], 2) + pow(colors[i][1] - colors[j][1], 2) + pow(colors[i][2] - colors[j][2], 2);
		float tmp2 = pow(colors[i][3] - colors[j][3], 2) + pow(colors[i][4] - colors[j][4], 2) + pow(colors[i][5] - colors[j][5], 2);
		return tmp1 + tmp2;
	}

	float randomNumber(float HI, float LO) {
		//Random number generation from stackoverflow https://stackoverflow.com/questions/686353/random-float-number-generation
		return LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
	}

	bool inConnections(Pair p) {
		for (int j = 0; j < NUM_OF_CONNECTIONS; ++j)
			if (p == connections[j])
				return true;
		return false;
	}

	void initVelocities() {
		for (int i = 0; i < NUM_OF_VERTICES; ++i)
			velocities[i] = vec3(0, 0, 0);
	}

	void correctCoordinates() {
		for (int i = 0; i < NUM_OF_CONNECTIONS; ++i) {
			float tmpX = fabs(vertices[i].x) <= FLT_MIN ? 0.000173 : vertices[i].x;
			float tmpY = fabs(vertices[i].y) <= FLT_MIN ? 0.000173 : vertices[i].y;
			if (fabs(vertices[i].x) <= FLT_MIN && vertices[i].x < 0) { tmpX *= -1; }
			if (fabs(vertices[i].y) <= FLT_MIN && vertices[i].y < 0) { tmpY *= -1; }
			vertices[i] = vec3(tmpX, tmpY, sqrtf(tmpX * tmpX + tmpY * tmpY + 1));
		}
	}

	void updateConnected() {
		int idxC = 0;
		for (int i = 0; i < NUM_OF_CONNECTIONS; ++i) {
			connected[idxC++] = vertices[connections[i].x];
			connected[idxC++] = vertices[connections[i].y];
		}
	}

	float hyperbolicDistance(vec3 a, vec3 b) {
		float lorentzProduct = a.x * b.x + a.y * b.y - a.z * b.z;
		if (isnan(acoshf(-lorentzProduct))) return 0;
		else return acoshf(-lorentzProduct);
	}

	void shiftVertices(vec3 from, vec3 to) {
		for (int i = 0; i < NUM_OF_VERTICES; ++i)
			shiftVertex(from, to, vertices[i]);
	}

	void shiftVertex(vec3 from, vec3 to, vec3& vertex) {
		float shiftVectorDistance = hyperbolicDistance(to, from);
		vec3 shiftVectorVelocity = (to - from * coshf(shiftVectorDistance)) / sinhf(shiftVectorDistance);
		vec3 mirrorPoint1 = from * coshf(shiftVectorDistance * 0.25) + shiftVectorVelocity * sinhf(shiftVectorDistance * 0.25);
		vec3 mirrorPoint2 = from * coshf(shiftVectorDistance * 0.75) + shiftVectorVelocity * sinhf(shiftVectorDistance * 0.75);
		float mp1Distance = hyperbolicDistance(mirrorPoint1, vertex);
		vec3 mp1Velocity = (mirrorPoint1 - vertex * coshf(mp1Distance)) / sinhf(mp1Distance);
		vec3 vertexMirrored1 = vertex * coshf(2 * mp1Distance) + mp1Velocity * sinhf(2 * mp1Distance);
		float mp2Distance = hyperbolicDistance(mirrorPoint2, vertexMirrored1);
		vec3 mp2Velocity = (mirrorPoint2 - vertexMirrored1 * coshf(mp2Distance)) / sinhf(mp2Distance);
		vec3 vertexMirrored2 = vertexMirrored1 * coshf(2 * mp2Distance) + mp2Velocity * sinhf(2 * mp2Distance);
		vertex = vertexMirrored2;
		correctCoordinates();
	}

	bool isConnected(const vec3 a, const vec3 b) {
		for (int l = 0; l < NUM_OF_CONNECTIONS; ++l) {
			const int idx1 = connections[l].x;
			const int idx2 = connections[l].y;
			if ((vertices[idx1].x == a.x && vertices[idx1].y == a.y && vertices[idx2].x == b.x && vertices[idx2].y == b.y) ||
				(vertices[idx1].x == b.x && vertices[idx1].y == b.y && vertices[idx2].x == a.x && vertices[idx2].y == a.y)) {
				return true;
			}
		}
		return false;
	}

	void kMeansClustering() {
		for (int i = 0; i < NUM_OF_VERTICES; ++i) {
			int M = 0; vec3 R(0, 0, 0);
			for (int j = 0; j < NUM_OF_VERTICES; ++j) {
				if (i == j) continue;
				if (isConnected(vertices[i], vertices[j])) {
					R = R + vertices[j];
					M++;
				}
				else {
					R = R - vertices[j];
					M--;
				}
			}
			R = R / M;
			R.z = sqrtf(R.x * R.x + R.y * R.y + 1);
			vertices[i] = R;
		}
		correctCoordinates();
	}

	void handleForceErrors(vec3& F) {
		if (fabs(F.x) < FLT_MIN || isinf(F.x) || isnan(F.x)) F.x = 0.0;
		if (fabs(F.y) < FLT_MIN || isinf(F.y) || isnan(F.y)) F.y = 0.0;
		if (fabs(F.z) < FLT_MIN || isinf(F.z) || isnan(F.z)) F.z = 0.0;
	}

	vec3 Fe(vec3 from, vec3 to) {
		float dist = hyperbolicDistance(from, to);
		float optDist = 0.3;
		float func = 20 * (dist - optDist) / (dist);
		if (func < -1) func = -1;
		vec3 Fe = (to - from * coshf(dist)) / sinhf(dist) * func;
		handleForceErrors(Fe);
		return Fe;
	}

	vec3 Fn(vec3 from, vec3 to) {
		float dist = hyperbolicDistance(from, to);
		float optDist = 0.3;
		float func = 5 * ((-1.0) * optDist) / pow(dist,3);
		if (func < -1) func = -1;
		vec3 Fn = (to - from * coshf(dist)) / sinhf(dist) * func;
		handleForceErrors(Fn);
		return Fn;
	}

	vec3 Fo(vec3 from) {
		float dist = hyperbolicDistance(from, vec3(0, 0, 1));
		float func = 10   * exp(1.5 * dist - 3);
		vec3 Fo = (vec3(0, 0, 1) - from * coshf(hyperbolicDistance(from, vec3(0, 0, 1)))) / sinhf(hyperbolicDistance(from, vec3(0, 0, 1))) * func;
		handleForceErrors(Fo);
		return Fo;
	}

	void dynamicSimulation(float dt, float simulationCycle) {
		for (int i = 0; i < NUM_OF_VERTICES; ++i) {
			vec3 Fi = vec3(0, 0, 0);
			for (int j = 0; j < NUM_OF_VERTICES; ++j) {
				if (i == j) continue;
				if (isConnected(vertices[i], vertices[j])) { Fi = Fi + Fe(vertices[i], vertices[j]); }
				else { Fi = Fi + Fn(vertices[i], vertices[j]); }
			}
			Fi = Fi + Fo(vertices[i]);
			Fi = Fi - velocities[i] * rho;

			velocities[i] = (velocities[i] + Fi / m * dt) * damping;
			float dist = length(velocities[i]) * dt;

			vec3 newVertex = vertices[i] * coshf(dist) + normalize(velocities[i]) * sinhf(dist);
			vec3 newVelocity = -length(velocities[i]) * normalize((vertices[i] - newVertex * coshf(dist)) / sinhf(dist));
			vertices[i] = newVertex;
			velocities[i] = newVelocity;
		}
		correctCoordinates();
	}

	void controlSimulation() {
		if (startClustering) {
			kMeansClustering();
			draw();
			startClustering = false;
		}
		if (startDynamicSimulation && simulationCycle < T) {
			dynamicSimulation(dt, simulationCycle);
			if (drawCycle % drawCount == 0) draw();
			simulationCycle += dt;
			++drawCycle;
		}
		else if (simulationCycle >= T) {
			simulationCycle = 0;
			startDynamicSimulation = false;
			draw();
		}
	}

	void create() {
		glGenVertexArrays(1, &vao0);
		glBindVertexArray(vao0);
		glGenBuffers(1, &vbo0);

		glGenVertexArrays(1, &vao1);
		glBindVertexArray(vao1);
		glGenBuffers(2, vbo1);
	}

	void draw() {
		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		int location = glGetUniformLocation(gpuProgram.getId(), "color");

		mat4 MVPTransform = { 1, 0, 0, 0,
							  0, 1, 0, 0,
							  0, 0, 1, 0,
							  0, 0, 0, 1 };
		gpuProgram.setUniform(MVPTransform, "MVP");

		correctCoordinates();
		updateConnected();
		gpuProgram.setUniform((int)false, "isTextured");
		glBindVertexArray(vao0);
		glBindBuffer(GL_ARRAY_BUFFER, vbo0);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vec3) * NUM_OF_CONNECTIONS * 2, connected, GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glUniform3f(location, 1.0f, 1.0f, 0.0f);
		glDrawArrays(GL_LINES, 0, NUM_OF_CONNECTIONS * 2);

		gpuProgram.setUniform((int)true, "isTextured");
		glBindVertexArray(vao1);

		for (int i = 0; i < NUM_OF_VERTICES; ++i) {
			const int nv = 50;
			std::vector<vec4> vector; //stores image

			float r = vertices[i].x / vertices[i].z;
			float g = vertices[i].y / vertices[i].z;
			float b = 1 / vertices[i].z;
			for (int i = 0; i < 100 * 100; ++i) {
				vec4 tmp(r + 0.35, g + 0.35, b + 0.35, 1);
				
				vector.push_back(tmp);
			}

			for (int x = 17; x < 83; ++x) {
				for (int y = 17; y < 83; ++y) {
					vector.at(x * 100 + y) = vec4(1,1,1,1);
				}
			}

			for (int x = 25; x < 75; ++x) {
				for (int y = 25; y < 75; ++y) {
					if (y < x - 5) {
						vector.at(x * 100 + y) = vec4(colors[i][0], colors[i][1], colors[i][2], 1);
					}
					if (y > x + 5) {
						vector.at(x * 100 + y) = vec4(colors[i][3], colors[i][4], colors[i][5], 1);
					}
				}
			}

			Texture txt(100, 100, vector); //texture

			gpuProgram.setUniform(txt, "texturer");

			vec2 uvs[50];
			for (int i = 0; i < nv; ++i) {
				float fi = i * 2.0 * M_PI / nv;
				uvs[i] = vec2(0.5, 0.5) + vec2(cosf(fi) * 0.5, sinf(fi) * 0.5);
			}

			glBindBuffer(GL_ARRAY_BUFFER, vbo1[1]);
			glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_DYNAMIC_DRAW);
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);

			vec3 circleVertices[nv];
			for (int j = 0; j < nv; ++j) {
				float fi = j * 2.0 * M_PI / nv;
				vec3 tmp = vertices[i];
				float r = 0.04;
				float shiftX = cosf(fi) * r;
				float shiftY = sinf(fi) * r;
				shiftVertex(vec3(0, 0, 1), vec3(shiftX, shiftY, sqrtf(shiftX * shiftX + shiftY * shiftY + 1)), tmp);
				circleVertices[j] = tmp;
			}
			glBindBuffer(GL_ARRAY_BUFFER, vbo1[0]);
			glBufferData(GL_ARRAY_BUFFER, sizeof(vec3) * nv, circleVertices, GL_DYNAMIC_DRAW);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
			glUniform3f(location, vertices[i].x / vertices[i].z, vertices[i].y / vertices[i].z, 1 / vertices[i].z);
			glDrawArrays(GL_TRIANGLE_FAN, 0, nv);
		}

		glutSwapBuffers();
	}

};

Graph graph;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glPointSize(8); glLineWidth(1);
	graph.create();

	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

void onDisplay() {
	graph.draw();
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd' || key == 'D') {
		graph = Graph();
		graph.create();
		graph.draw();
		glutPostRedisplay();
	}// if d, invalidate display, i.e. redraw
	if (key == ' ') {
		graph.initVelocities();
		graph.simulationCycle = 0;
		graph.drawCycle = 0;
		graph.startClustering = true;
		graph.startDynamicSimulation = true;
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

vec2 startPoint(0,0);
vec2 endPoint(0,0);
int mouseBtn = -1;

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	
	if (mouseBtn == GLUT_RIGHT_BUTTON) {
		startPoint = endPoint;
		endPoint = vec2(cX, cY);

		float tmpX = fabs((endPoint - startPoint).x) <= FLT_MIN ? 0.000173 : (endPoint - startPoint).x;
		float tmpY = fabs((endPoint - startPoint).y) <= FLT_MIN ? 0.000173 : (endPoint - startPoint).y;
		if (fabs((endPoint - startPoint).x) <= FLT_MIN && (endPoint - startPoint).x < 0) { tmpX *= -1; }
		if (fabs((endPoint - startPoint).y) <= FLT_MIN && (endPoint - startPoint).y < 0) { tmpY *= -1; }
		
		vec2 shiftPoint2D(tmpX, tmpY);
		//Beltrami-Klein inverse
		float winv = sqrtf(1 - shiftPoint2D.x * shiftPoint2D.x - shiftPoint2D.y * shiftPoint2D.y);
		vec3 shiftPointHyperbolic = vec3(shiftPoint2D.x / winv, shiftPoint2D.y / winv, 1 / winv);
		
		graph.shiftVertices(vec3(0, 0, 1),shiftPointHyperbolic);
		graph.draw();
	}
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	
	mouseBtn = -1;
	if (button == GLUT_RIGHT_BUTTON) {
		if (state == GLUT_DOWN) { mouseBtn = GLUT_RIGHT_BUTTON;  startPoint = vec2(cX, cY); endPoint = vec2(cX, cY); }
	}
}

void onIdle() {
	graph.controlSimulation();
}
