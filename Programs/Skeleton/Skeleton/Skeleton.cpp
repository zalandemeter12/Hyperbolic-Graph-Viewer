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
// Nev    : Demeter Zalán
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



GPUProgram gpuProgram;	// vertex and fragment shaders

class Graph {
	static const int NUM_OF_VERTICES = 50;
	static const int NUM_OF_CONNECTIONS = 61;

	unsigned int vao0;
	unsigned int vao1;
	float sx, sy;		// scaling
	vec2 wTranslate;	// translation
	float phi;			// angle of rotation
public:
	vec4 vertices[NUM_OF_VERTICES];
	vec2 connections[NUM_OF_CONNECTIONS];

	Graph() {
		for (int i = 0; i < NUM_OF_VERTICES; ++i) {
			//Random number generation from stackoverflow
			//https://stackoverflow.com/questions/686353/random-float-number-generation
			float MAX = 3.5;
			float MIN = -3.5;
			float x = MIN + (float)(rand()) / ((float)(RAND_MAX / (MAX - MIN)));
			float y = MIN + (float)(rand()) / ((float)(RAND_MAX / (MAX - MIN)));
			vertices[i] = vec4(x, y, 0, sqrt(x * x + y * y + 1));
		}

		int i = 0;
		int idx = 0;
		while (i < NUM_OF_CONNECTIONS) {
			int idx1 = rand() % NUM_OF_VERTICES, idx2 = rand() % NUM_OF_VERTICES;
			if (idx1 == idx2) continue;
			bool inConnections = false;
			for (int j = 0; j < NUM_OF_CONNECTIONS; ++j) {
				if ((connections[j].x == idx1 && connections[j].y == idx2) || (connections[j].x == idx2 && connections[j].y == idx1)) {
					inConnections = true;
					break;
				}
			}
			if (inConnections) continue;
			connections[idx++] = vec2(idx1, idx2);
			++i;
		}

		sx = 1; sy = 1;
		wTranslate = vec2(0, 0);
		phi = 0;
	}

	void Create() {
		glGenVertexArrays(1, &vao0);
		glBindVertexArray(vao0);
		unsigned int vbo0;
		glGenBuffers(1, &vbo0);
		glBindBuffer(GL_ARRAY_BUFFER, vbo0);
		int idxV = 0;
		float arrayV[NUM_OF_VERTICES * 2];
		for (int i = 0; i < NUM_OF_VERTICES; ++i) {
			arrayV[idxV++] = vertices[i].x / vertices[i].w;
			arrayV[idxV++] = vertices[i].y / vertices[i].w;
		}
		glBufferData(GL_ARRAY_BUFFER, sizeof(arrayV), arrayV, GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		glGenVertexArrays(1, &vao1);
		glBindVertexArray(vao1);
		unsigned int vbo1;
		glGenBuffers(1, &vbo1);
		glBindBuffer(GL_ARRAY_BUFFER, vbo1);
		int idxC = 0;
		float arrayC[NUM_OF_CONNECTIONS * 4];
		for (int i = 0; i < NUM_OF_CONNECTIONS; ++i) {
			arrayC[idxC++] = vertices[(const int)connections[i].x].x / vertices[(const int)connections[i].x].w;
			arrayC[idxC++] = vertices[(const int)connections[i].x].y / vertices[(const int)connections[i].x].w;
			arrayC[idxC++] = vertices[(const int)connections[i].y].x / vertices[(const int)connections[i].y].w;
			arrayC[idxC++] = vertices[(const int)connections[i].y].y / vertices[(const int)connections[i].y].w;
		}
		glBufferData(GL_ARRAY_BUFFER, sizeof(arrayC), arrayC, GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	mat4 M() {
		mat4 Mscale(sx, 0, 0, 0,
			0, sy, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 1); // scaling

		mat4 Mrotate(cosf(phi), sinf(phi), 0, 0,
			-sinf(phi), cosf(phi), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1); // rotation

		mat4 Mtranslate(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 0,
			wTranslate.x, wTranslate.y, 0, 1); // translation

		return Mscale * Mrotate * Mtranslate;	// model transformation
	}

	void Draw() {
		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT);

		mat4 MVPTransform = M();
		gpuProgram.setUniform(MVPTransform, "MVP");
		int location = glGetUniformLocation(gpuProgram.getId(), "color");

		glBindVertexArray(vao0);
		glUniform3f(location, 1.0f, 0.0f, 0.0f);
		glDrawArrays(GL_POINTS, 0, NUM_OF_VERTICES);

		glBindVertexArray(vao1);
		glUniform3f(location, 1.0f, 1.0f, 0.0f);
		glDrawArrays(GL_LINES, 0, NUM_OF_CONNECTIONS * 2);
		glutSwapBuffers();
	}

	bool isConnected(const vec4 a, const vec4 b) {
		bool connected = false;
		for (int l = 0; l < NUM_OF_CONNECTIONS; ++l) {
			const int idx1 = (const int)connections[l].x;
			const int idx2 = (const int)connections[l].y;
			if ((vertices[idx1].x == a.x && vertices[idx1].y == a.y && vertices[idx2].x == b.x && vertices[idx2].y == b.y) ||
				(vertices[idx1].x == b.x && vertices[idx1].y == b.y && vertices[idx2].x == a.x && vertices[idx2].y == a.y)) {
				connected = true;
				break;
			}
		}
		return connected;
	}
	int numOfConnections(vec4 a) {
		int c = 0;
		for (int i = 0; i < NUM_OF_CONNECTIONS; ++i)
			if (vertices[(const int)connections[i].x].x == a.x || vertices[(const int)connections[i].y].x == a.x)
				c++;
		return c;
	}

	int* bubbleSortIdx() {
		static int sorted[NUM_OF_VERTICES];
		for (int i = 0; i < NUM_OF_VERTICES; ++i) {
			sorted[i] = i;
		}
		for (int i = 0; i < NUM_OF_VERTICES - 1; i++) {
			for (int j = 0; j < NUM_OF_VERTICES - i - 1; j++) {
				if (numOfConnections(vertices[sorted[j]]) < numOfConnections(vertices[sorted[j + 1]])) {
					int t = sorted[j];
					sorted[j] = sorted[j + 1];
					sorted[j + 1] = t;
				}
			}
		}
		return sorted;
	}

	void kMeansClustering() {
		int* sorted = bubbleSortIdx();
		for (int i = 0; i < 50; ++i) {
			int M = 0;
			vec4 R = vec4(0, 0, 0, 0);
			for (int j = 0; j < 50; ++j) {
				if (sorted[i] == j) continue;
				if (isConnected(vertices[sorted[i]], vertices[j])) {
					R += (+1) * vertices[j];
					M++;
				}
				else {
					R += (-1) * vertices[j];
					M--;
				}
			}
			R = R / M;
			R.w = sqrt(R.x * R.x + R.y * R.y + 1.0);
			vertices[sorted[i]] = R;
		}
	}
};

Graph graph;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glPointSize(8); glLineWidth(1);
	graph.Create();
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

void onDisplay() {
	graph.Draw();
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
	if (key == ' ') {
		graph.kMeansClustering();
		graph.Create();
		graph.Draw();
	}
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
