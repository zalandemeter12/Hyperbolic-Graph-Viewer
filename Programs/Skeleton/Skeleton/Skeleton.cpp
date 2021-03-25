//=============================================================================================
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

#define FLT_MIN				1.175494351e-38F        
#define NUM_OF_VERTICES		50						
#define NUM_OF_CONNECTIONS	61						

const char * const vertexSource = R"(
	#version 330									
	precision highp float;							

	uniform mat4 MVP;								
	layout(location = 0) in vec3 vertexPosition;	
	layout(location = 1) in vec2 vertexUV;			

	out vec2 texCoord;								

	void main() {
		texCoord = vertexUV;
		gl_Position = vec4(vertexPosition.x/vertexPosition.z, vertexPosition.y/vertexPosition.z, 0, 1) * MVP;
	}
)";

const char* fragmentSource = R"(
	#version 330
	precision highp float;

	uniform int isTextured;
	uniform sampler2D textureUnit;
	uniform vec3 color;					
	in vec2 texCoord;					
	out vec4 outColor;					

	void main() {
		if (isTextured != 0)
			outColor = texture(textureUnit, texCoord);
		else
			outColor = vec4(color, 1);
	}
)";

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
	GPUProgram gpuProgram;
	unsigned int vao0, vao1;
	unsigned int vbo0, vbo1[2];
	Pair connections[NUM_OF_CONNECTIONS];
	vec3 vertices[NUM_OF_VERTICES];
	vec3 velocities[NUM_OF_VERTICES];
	vec3 colors[NUM_OF_VERTICES][2];
public:
	bool startClustering = false;
	bool startDynamicSimulation = false;
	float simulationCycle = 0;
	int drawCycle = 0;
	int drawCount = 10;
	float T = 15;
	float dt = 0.025;
	int rho = 2;
	int m = 1;
	float damping = 0.975;

	Graph() {
		for (int i = 0; i < NUM_OF_VERTICES; ++i) {
			float x = randomFloat(-1, 1);
			float y = randomFloat(-1, 1);
			vertices[i] = vec3(x, y, sqrtf(x * x + y * y + 1));
		}
		int i = 0;
		while (i < NUM_OF_CONNECTIONS) {
			int idx1 = rand() % NUM_OF_VERTICES, idx2 = rand() % NUM_OF_VERTICES;
			if (idx1 == idx2) continue;
			if (isConnected(Pair(idx1, idx2), connections)) continue;
			connections[i] = Pair(idx1, idx2);
			++i;
		}
		
		//Generating N visually distinct colors using the HSV color model from here:
		//https://en.wikipedia.org/wiki/HSL_and_HSV#HSV
		std::vector<vec3> tmpColors;
		int numOfColors = (NUM_OF_VERTICES / 4);
		for (float i = 0.0; i < 360.0; i += 360.0 / numOfColors) {
			int hue = i;
			float saturation = 0.8 + randomFloat(0,1) * 0.2;
			float value = 0.8 + randomFloat(0, 1) * 0.2;
			tmpColors.push_back(HSVtoRGB(hue, saturation, value));
		}
		Pair colorPairs[NUM_OF_VERTICES];
		int j = 0;
		while (j < NUM_OF_VERTICES) {
			int idx1 = rand() % numOfColors, idx2 = rand() % numOfColors;
			if (idx1 == idx2) continue;
			if (isConnected(Pair(idx1, idx2), colorPairs)) continue;
			colorPairs[j] = Pair(idx1, idx2);
			colors[j][0] = vec3(tmpColors[idx1].x, tmpColors[idx1].y, tmpColors[idx1].z);
			colors[j][1] = vec3(tmpColors[idx2].x, tmpColors[idx2].y, tmpColors[idx2].z);
			++j;
		}
	}

	//HSV to RGB conversion using the formula from here: 
	//https://www.rapidtables.com/convert/color/hsv-to-rgb.html
	vec3 HSVtoRGB(float H, float S, float V) {
		float C = S * V;
		float X = C * (1 - fabs(fmod(H / 60.0, 2) - 1));
		float m = V - C;
 		if (0 <= H && H < 60)    { return vec3((C + m), (X + m), (0 + m)); }
		if (60 <= H && H < 120)  { return vec3((X + m), (C + m), (0 + m)); }
		if (120 <= H && H < 180) { return vec3((0 + m), (C + m), (X + m)); }
		if (180 <= H && H < 240) { return vec3((0 + m), (X + m), (C + m)); }
		if (240 <= H && H < 300) { return vec3((X + m), (0 + m), (C + m)); }
		if (300 <= H && H < 360) { return vec3((C + m), (0 + m), (X + m)); }
	}

	float randomFloat(float LO, float HI) {
		//Random float generation from here:
		//https://stackoverflow.com/questions/686353/random-float-number-generation
		return LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
	}

	bool isConnected(Pair p, Pair list[]) {
		for (int i = 0; i < NUM_OF_CONNECTIONS; ++i)
			if (p == list[i])
				return true;
		return false;
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

	float hyperbolicDistance(vec3 a, vec3 b) {
		float lorentzProduct = a.x * b.x + a.y * b.y - a.z * b.z;
		if (isnan(acoshf(-lorentzProduct))) return 0;
		else return acoshf(-lorentzProduct);
	}

	vec3 hyperbolicVelocity(vec3 from, vec3 to, float distance) {
		return (to - from * coshf(distance)) / sinhf(distance);
	}

	vec3 hyperbolicMovement(vec3 from, float distance, vec3 velocity) {
		return from * coshf(distance) + velocity * sinhf(distance);
	}

	void shiftVertices(vec3 from, vec3 to) {
		for (int i = 0; i < NUM_OF_VERTICES; ++i)
			shiftVertex(from, to, vertices[i]);
	}

	void shiftVertex(vec3 from, vec3 to, vec3& vertex) {
		float shiftVectorDistance =	hyperbolicDistance(to, from);
		vec3 shiftVectorVelocity =	hyperbolicVelocity(from, to, shiftVectorDistance);
		vec3 mirrorPoint1 = hyperbolicMovement(from, shiftVectorDistance * 0.25, shiftVectorVelocity);
		vec3 mirrorPoint2 = hyperbolicMovement(from, shiftVectorDistance * 0.75, shiftVectorVelocity);
		float mp1Distance = hyperbolicDistance(mirrorPoint1, vertex);
		vec3 mp1Velocity = hyperbolicVelocity(vertex, mirrorPoint1, mp1Distance);
		vec3 vertexMirrored1 = hyperbolicMovement(vertex, 2 * mp1Distance, mp1Velocity);
		float mp2Distance = hyperbolicDistance(mirrorPoint2, vertexMirrored1);
		vec3 mp2Velocity = hyperbolicVelocity(vertexMirrored1, mirrorPoint2, mp2Distance);
		vertex = hyperbolicMovement(vertexMirrored1, 2 * mp2Distance, mp2Velocity);
		correctCoordinates();
	}

	int numOfConnections(unsigned int idx) {
		int c = 0;
		for (int i = 0; i < NUM_OF_CONNECTIONS; ++i)
			if (isConnected(Pair(idx, i), connections))
				c++;
		return c;
	}

	void kMeansClustering() {
		for (int i = 0; i < NUM_OF_VERTICES; ++i) {
			//Center of mass calculation with the formula from here:
			//https://en.wikipedia.org/wiki/Center_of_mass
			vec3 R(0, 0, 0);
			for (int j = 0; j < NUM_OF_VERTICES; ++j) {
				if (i == j) continue;
				if (isConnected(Pair(i,j),connections)) R = R + vertices[j];
				else R = R - vertices[j];
			}
			vertices[i] = R / (2 * numOfConnections(i) - NUM_OF_VERTICES + 1);
			vertices[i].z = sqrtf(R.x * R.x + R.y * R.y + 1);
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
		if (func < -2) func = -2;
		vec3 Fe = normalize(hyperbolicVelocity(from, to, dist)) * func;
		handleForceErrors(Fe);
		return Fe;
	}

	vec3 Fn(vec3 from, vec3 to) {
		float dist = hyperbolicDistance(from, to);
		float optDist = 0.3;
		float func = 5 * ((-1.0) * optDist) / pow(dist,3);
		if (func < -2) func = -2;
		vec3 Fn = normalize(hyperbolicVelocity(from, to, dist)) * func;
		handleForceErrors(Fn);
		return Fn;
	}

	vec3 Fo(vec3 from) {
		float dist = hyperbolicDistance(from, vec3(0, 0, 1));
		float func = 20 * exp(1.5 * dist - 3);
		vec3 Fo = normalize(hyperbolicVelocity(from, vec3(0, 0, 1), dist)) * func;
		handleForceErrors(Fo);
		return Fo;
	}

	void dynamicSimulation() {
		for (int i = 0; i < NUM_OF_VERTICES; ++i) {
			vec3 Fi = Fo(vertices[i]);
			for (int j = 0; j < NUM_OF_VERTICES; ++j) {
				if (i == j) continue;
				if (isConnected(Pair(i, j), connections)) { Fi = Fi + Fe(vertices[i], vertices[j]); }
				else { Fi = Fi + Fn(vertices[i], vertices[j]); }
			}		
			Fi = Fi - velocities[i] * rho;

			velocities[i] = (velocities[i] + Fi / m * dt) * damping;
			float dist = length(velocities[i]) * dt;

			vec3 newVertex = hyperbolicMovement(vertices[i], dist, normalize(velocities[i]));
			vec3 newVelocity = -length(velocities[i]) * normalize(hyperbolicVelocity(newVertex, vertices[i], dist));
			vertices[i] = newVertex;
			velocities[i] = newVelocity;
		}
		correctCoordinates();
	}

	void initSimulation() {
		for (int i = 0; i < NUM_OF_VERTICES; ++i)
			velocities[i] = vec3(0, 0, 0);
		simulationCycle = 0;
		drawCycle = 0;
		startClustering = true;
		startDynamicSimulation = true;
	}

	void controlSimulation() {
		if (startClustering) {
			kMeansClustering();
			draw();
			startClustering = false;
		}
		if (startDynamicSimulation && simulationCycle < T) {
			dynamicSimulation();
			if (drawCycle % drawCount == 0) draw();
			simulationCycle += dt;
			++drawCycle;
		}
		if (simulationCycle >= T) {
			simulationCycle = 0;
			startDynamicSimulation = false;
			draw();
		}
	}

	void create() {
		glViewport(0, 0, windowWidth, windowHeight);
		glGenVertexArrays(1, &vao0);
		glBindVertexArray(vao0);
		glGenBuffers(1, &vbo0);
		glGenVertexArrays(1, &vao1);
		glBindVertexArray(vao1);
		glGenBuffers(2, vbo1);
		gpuProgram.create(vertexSource, fragmentSource, "outColor");
	}

	void draw() {
		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glPointSize(8);
		glLineWidth(2);

		mat4 MVPTransform = { 1, 0, 0, 0,
							  0, 1, 0, 0,
							  0, 0, 1, 0,
							  0, 0, 0, 1 };
		gpuProgram.setUniform(MVPTransform, "MVP");

		int idx = 0;
		vec3 connected[NUM_OF_CONNECTIONS * 2];
		for (int i = 0; i < NUM_OF_CONNECTIONS; ++i) {
			connected[idx++] = vertices[connections[i].x];
			connected[idx++] = vertices[connections[i].y];
		}

		gpuProgram.setUniform((int)false, "isTextured");
		glBindVertexArray(vao0);
		glBindBuffer(GL_ARRAY_BUFFER, vbo0);
		glBufferData(GL_ARRAY_BUFFER, sizeof(connected), connected, GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, 0.6f, 0.6f, 0.6f);
		glDrawArrays(GL_LINES, 0, NUM_OF_CONNECTIONS * 2);
		
		for (int i = 0; i < NUM_OF_VERTICES; ++i) {
			const int nv = 50;
			vec3 circleVertices[nv];
			vec2 uvs[nv];
			for (int j = 0; j < nv; ++j) {
				float fi = j * 2.0 * M_PI / nv;
				vec3 tmp = vertices[i];
				float r = 0.04;
				float shiftX = cosf(fi) * r;
				float shiftY = sinf(fi) * r;
				shiftVertex(vec3(0, 0, 1), vec3(shiftX, shiftY, sqrtf(shiftX * shiftX + shiftY * shiftY + 1)), tmp);
				circleVertices[j] = tmp;
				uvs[j] = vec2(0.5, 0.5) + vec2(cosf(fi) / 2, sinf(fi) / 2);
			}

			int width = 100, height = 100;
			std::vector<vec4> image(width * height);
			for (int j = 0; j < width * height; ++j)
				image.at(j) = vec4((vertices[i].x / vertices[i].z) + 0.35, (vertices[i].y / vertices[i].z) + 0.35, (1 / vertices[i].z) + 0.35, 1);
			for (int x = width * 0.17; x < width * 0.83; ++x)
				for (int y = height * 0.17; y < height * 0.83; ++y)
					image.at(x * width + y) = vec4(1,1,1,1);
			for (int x = width * 0.25; x < width * 0.75; ++x) {
				for (int y = height * 0.25; y < height * 0.75; ++y) {
					if (y < x - 5)
						image.at(x * width + y) = vec4(colors[i][0].x, colors[i][0].y, colors[i][0].z, 1);
					if (y > x + 5)
						image.at(x * width + y) = vec4(colors[i][1].x, colors[i][1].y, colors[i][1].z, 1);
				}
			}

			gpuProgram.setUniform((int)true, "isTextured");
			Texture texture(width, height, image);
			gpuProgram.setUniform(texture, "textureUnit");
			glBindVertexArray(vao1);
			glBindBuffer(GL_ARRAY_BUFFER, vbo1[0]);
			glBufferData(GL_ARRAY_BUFFER, sizeof(circleVertices), circleVertices, GL_DYNAMIC_DRAW);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
			glBindBuffer(GL_ARRAY_BUFFER, vbo1[1]);
			glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_DYNAMIC_DRAW);
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
			glDrawArrays(GL_TRIANGLE_FAN, 0, nv);
		}
		glutSwapBuffers();
	}
};

Graph graph;

void onInitialization() {
	graph.create();
}

void onDisplay() {
	graph.draw();
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd' || key == 'D') {
		graph = Graph();
		graph.create();
		graph.draw();
		glutPostRedisplay();
	}
	if (key == ' ') {
		graph.initSimulation();
	}
}

vec2 startPoint(0,0);
vec2 endPoint(0,0);
int mouseBtn = -1;

void onMouseMotion(int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
	
	if (mouseBtn == GLUT_RIGHT_BUTTON) {
		startPoint = endPoint;
		endPoint = vec2(cX, cY);

		float tmpX = fabs((endPoint - startPoint).x) <= FLT_MIN ? 0.000173 : (endPoint - startPoint).x;
		float tmpY = fabs((endPoint - startPoint).y) <= FLT_MIN ? 0.000173 : (endPoint - startPoint).y;
		if (fabs((endPoint - startPoint).x) <= FLT_MIN && (endPoint - startPoint).x < 0) { tmpX *= -1; }
		if (fabs((endPoint - startPoint).y) <= FLT_MIN && (endPoint - startPoint).y < 0) { tmpY *= -1; }
		
		vec2 shiftPoint2D(tmpX, tmpY);
		float winv = sqrtf(1 - shiftPoint2D.x * shiftPoint2D.x - shiftPoint2D.y * shiftPoint2D.y);
		vec3 shiftPointHyperbolic = vec3(shiftPoint2D.x / winv, shiftPoint2D.y / winv, 1 / winv);
		
		graph.shiftVertices(vec3(0, 0, 1),shiftPointHyperbolic);
		graph.draw();
	}
}

void onMouse(int button, int state, int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
	
	mouseBtn = -1;
	if (button == GLUT_RIGHT_BUTTON) {
		if (state == GLUT_DOWN) { mouseBtn = GLUT_RIGHT_BUTTON;  startPoint = vec2(cX, cY); endPoint = vec2(cX, cY); }
	}
}

void onIdle() {
	graph.controlSimulation();
}
