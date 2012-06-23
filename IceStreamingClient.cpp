// Ice
#include <ice/ice.h>

// Cuda
#include <vector_functions.h>

// OpenGL Graphics Includes
#include <GL/glew.h>
#include <GL/freeglut.h>

// Includes
#include <memory>
#include <time.h>
#include <iostream>
#include <cassert>

// Project
#include "IIceStreamer.h"

// Ice
::Ice::CommunicatorPtr communicator;
::Streamer::BitmapProviderPrx bitmapProvider;

// General Settings
const long TARGET_FPS = 200;
const long REFRESH_DELAY = 1000/TARGET_FPS; //ms

// GPU
int platform     = 0;
int device       = 0;

// Rendering window vars
const unsigned int draft        = 1;
unsigned int window_width       = 256;
unsigned int window_height      = 256;
const unsigned int window_depth = 4;

// ----------------------------------
// Scene
// ----------------------------------------------------------------------
const float gRoomWidth  = 2000.f;
const float gRoomHeight = 1000.f;
const float gRoomDepth  = 2000.f;
float gTransparentColor = 0.01f;
int   currentMaterial   = 0;

int gMaterials[2];

int nbPrimitives = 0;
int nbLamps      = 0;
int nbMaterials  = 0;
int nbTextures   = 20;

// Camera
float4 gViewPos;
float4 gViewDir;
float4 gViewAngles;
float4 gPreviewViewPos;
float4 lampPosition;

// materials
float gReflection   = 0.f;
float gRefraction   = 1.f;
float gTransparency = 0.f;
float gSpecValue  = 1.f;
float gSpecPower  = 100.f; 
float gSpecCoef   = 1.f;
float gInnerIllumination = 0.f;

#ifdef USE_KINECT
const float gSkeletonSize      = 200.0;
const float gSkeletonThickness =  20.0;
#endif // USE_KINECT

// --------------------------------------------------------------------------------
// OpenGL
// --------------------------------------------------------------------------------
GLubyte* ubImage;

// GL functionality
void initgl(int argc, char** argv);
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion( int x, int y );
void timerEvent( int value );
void createScene( int platform, int device );

// Helpers
void TestNoGL();
void Cleanup(int iExitCode);
void (*pCleanup)(int) = &Cleanup;

// Sim and Auto-Verification parameters 
float anim = 0.f;
float gDepthOfField  = 0.f;
bool bNoPrompt = false;  

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;

// Raytracer Module
unsigned int* uiOutput = NULL;

float getRandomValue( int range, int safeZone, bool allowNegativeValues = true )
{
	float value( static_cast<float>(rand()%range) + safeZone);
	if( allowNegativeValues ) 
	{
		value *= (rand()%2==0)? -1 : 1;
	}
	return value;
}

void idle()
{
}

void cleanup()
{
}

void animateSkeleton()
{
#if USE_KINECT
   long hr = gpuKernel->updateSkeletons(
      0.f, gSkeletonSize-500.f, 100.f,          // Position
      gSkeletonSize,                // Skeleton size
      gSkeletonThickness,         2, // Default material
      gSkeletonThickness*2.0f,    1, // Head size and material
      gSkeletonThickness*1.5f,    10, // Hands size and material
      gSkeletonThickness*1.8f,    10  // Feet size and material
      );
   if( hr == S_OK )
   {
      float4 position;
      const float size = 1500.f;

      // Left Hand
      if( gpuKernel->getSkeletonPosition(NUI_SKELETON_POSITION_HAND_LEFT, position) )
      {
         for( int i(0); i<6; ++i )
         {
            float x,y,z,w;
            gpuKernel->getPrimitiveCenter(leftPrimitiveId-i,x,y,z,w);
            gpuKernel->setPrimitiveCenter(leftPrimitiveId-i,x,position.y*size,z,w );
         }
      }

      // Right Hand
      if( gpuKernel->getSkeletonPosition(NUI_SKELETON_POSITION_HAND_RIGHT, position) )
      {
         for( int i(0); i<6; ++i )
         {
            float x,y,z,w;
            gpuKernel->getPrimitiveCenter(rightPrimitiveId-i,x,y,z,w);
            gpuKernel->setPrimitiveCenter(rightPrimitiveId-i,x,position.y*size,z,w );
         }
      }

      // Head
      if( gpuKernel->getSkeletonPosition(NUI_SKELETON_POSITION_HEAD, position) )
      {

         gViewAngles.y = -0.8f*asin( 
            position.x - 
            gPreviewViewPos.x );
         gViewAngles.x = 0.8f*asin( 
            position.y - 
            gPreviewViewPos.y );

         gPreviewViewPos.x = position.x;
         gPreviewViewPos.y = position.y;

         gViewPos.y -= 400.f;
         gpuKernel->setCamera( gViewPos, gViewDir, gViewAngles );
         gViewPos.y += 400.f;
      }
   }
#endif // USE_KINEXT
}

/*
--------------------------------------------------------------------------------
setup the window and assign callbacks
--------------------------------------------------------------------------------
*/
void initgl( int argc, char **argv )
{
	size_t len(window_width*window_height*window_depth);
	ubImage = new GLubyte[len];
	memset( ubImage, 0, len ); 

	glutInit(&argc, (char**)argv);
	glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );

	glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - window_width/2, glutGet(GLUT_SCREEN_HEIGHT)/2 - window_height/2);

	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Cuda Raytracer");

	glutDisplayFunc(display);       // register GLUT callback functions
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutTimerFunc(REFRESH_DELAY,timerEvent,REFRESH_DELAY);
	return;
}

void TexFunc(void)
{
	glEnable(GL_TEXTURE_2D);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);

	glTexImage2D(GL_TEXTURE_2D, 0, 3, window_width, window_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, ubImage);

	glBegin(GL_QUADS);
	glTexCoord2f(1.0, 1.0);
	glVertex3f(-1.0, 1.0, 0.0);
	glTexCoord2f(0.0, 1.0);
	glVertex3f( 1.0, 1.0, 0.0);
	glTexCoord2f(0.0, 0.0);
	glVertex3f( 1.0,-1.0, 0.0);
	glTexCoord2f(1.0, 0.0);
	glVertex3f(-1.0,-1.0, 0.0);
	glEnd();

	glDisable(GL_TEXTURE_2D);
}

// Display callback
//*****************************************************************************
void display()
{
	// clear graphics
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	TexFunc();
	glFlush();

	glutSwapBuffers();
}

void timerEvent(int value)
{
   try 
   {
      Streamer::bytes bitmap = bitmapProvider->getBitmap( 
         gViewPos.x, gViewPos.y, gViewPos.z, 
         gViewDir.x, gViewDir.y, gViewDir.z, 
         gViewAngles.x, gViewAngles.y, gViewAngles.z,
         anim, gDepthOfField, gTransparentColor );
      int c(0);
      for( int i(54); i<bitmap.size()-3; i+=3) {
         ubImage[c+0] = bitmap[i+2];
         ubImage[c+1] = bitmap[i+1];
         ubImage[c+2] = bitmap[i+0];
         ubImage[c+3] = 0;
         c+=4;
      }
   }
   catch(const Ice::Exception& e)
   {
      std::cerr << e.ice_stackTrace() << std::endl;
   }
   catch( ... ) 
   {
      std::cerr << "Unknown exception" << std::endl;
   }
	
	glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);

	anim += 0.02f;
}

// Keyboard events handler
//*****************************************************************************
void keyboard(unsigned char key, int x, int y)
{
	srand(static_cast< unsigned int>(time(NULL))); 

	switch(key) 
	{
	case '\033': 
	case '\015': 
	case 'X':    
	case 'x':    
		{
			// Cleanup up and quit
			bNoPrompt = true;
			Cleanup(EXIT_SUCCESS);
			break;
		}
	}
}

// Mouse event handlers
//*****************************************************************************
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN) 
	{
		mouse_buttons |= 1<<button;
	} 
	else 
	{
		if (state == GLUT_UP) 
		{
			mouse_buttons = 0;
		}
	}
	mouse_old_x = x;
	mouse_old_y = y;
	//gViewAngles.x = 0.f;
	//gViewAngles.y = 0.f;
}

void motion(int x, int y)
{
	switch( mouse_buttons ) 
	{
	case 1:

      if( glutGetModifiers() != GLUT_ACTIVE_CTRL ) 
      {
         // Move gViewPos position along the Z axis
         gViewPos.z += 2*(mouse_old_y-y);
         if( glutGetModifiers() != GLUT_ACTIVE_SHIFT ) 
         {
            gViewDir.z += 2*(mouse_old_y-y);
         }
      }
      else
      {
         gDepthOfField += 10*(mouse_old_y-y);
      }
		break;
	case 2:
		// Rotates the scene around X and Y axis
		gViewAngles.y += -asin( (mouse_old_x-x) / 100.f );
		gViewAngles.x += asin( (mouse_old_y-y) / 100.f );
		break;
	case 4:
		// Move gViewPos postion along X and Y axis
		gViewPos.x += (mouse_old_x-x);
		gViewPos.y += (mouse_old_y-y);
		gViewDir.x += (mouse_old_x-x);
		gViewDir.y += (mouse_old_y-y);
		break;
	}
	mouse_old_x = x;
	mouse_old_y = y;

}

// Function to clean up and exit
//*****************************************************************************
void Cleanup(int iExitCode)
{
	// Cleanup allocated objects
	std::cout << "\nStarting Cleanup...\n\n" << std::endl;

	if( ubImage ) delete [] ubImage;
   communicator->destroy();

	exit (iExitCode);
}

void main( int argc, char* argv[] )
{
	if( argc == 5 ) {
		std::cout << argv[1] << std::endl;
		sscanf_s( argv[1], "%d", &platform );
		sscanf_s( argv[2], "%d", &device );
		sscanf_s( argv[3], "%d", &window_width );
		sscanf_s( argv[4], "%d", &window_height );
	}

   // Camera initialization
   gViewPos.x =     0.f;
   gViewPos.y =     0.f;
   gViewPos.z = -1000.f;

   gViewDir.x = 0.f;
   gViewDir.y = 0.f;
   gViewDir.z = 0.f;

   gViewAngles.x = 0.f;
   gViewAngles.y = 0.f;
   gViewAngles.z = 0.f;

   try 
   {
      communicator = Ice::initialize(argc, argv);
      std::cout << "Connecting to Ice server ..." << std::endl;

      std::string connectionString;
      connectionString += "IceStreamer:tcp -p 10000 -z";
      connectionString += " -h ";
      if( argc == 2 ) 
      {
         connectionString += argv[1];
      }
      else
      {
         connectionString += "localhost";
      }

      std::cout << "Connection string: " << connectionString << std::endl;
      bitmapProvider = ::Streamer::BitmapProviderPrx::checkedCast(
         communicator->stringToProxy(connectionString));

      std::cout << "helloWorld: " << bitmapProvider->helloWorld("Say something") << std::endl;

      // First initialize OpenGL context, so we can properly set the GL for CUDA.
      // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
      initgl( argc, argv );

      atexit(cleanup);
      glutMainLoop();
   }
   catch(const Ice::Exception& e)
   {
      std::cerr << e.ice_stackTrace() << std::endl;
   }
   catch( ... ) 
   {
      std::cerr << "Unknown exception" << std::endl;
   }

	// Normally unused return path
	Cleanup(EXIT_SUCCESS);
}
