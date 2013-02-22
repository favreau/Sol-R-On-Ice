/* 
* GPU Raytracer
* Copyright (C) 2011-2012 Cyrille Favreau <cyrille_favreau@hotmail.com>
*
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Library General Public
* License as published by the Free Software Foundation; either
* version 2 of the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Library General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>. 
*/

/*
* Author: Cyrille Favreau <cyrille_favreau@hotmail.com>
*
*/

#define _CRT_SECURE_NO_WARNINGS

// Ice
#include <Ice/Ice.h>

// OpenGL Graphics Includes
#include <GL/glew.h>
#include <GL/freeglut.h>

// Includes
#include <memory>
#include <time.h>
#include <iostream>
#include <cassert>
#include <vector>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <vector_types.h>

// Project
#include "IIceStreamer.h"

// Ice
::Ice::CommunicatorPtr gCommunicator;
::IceStreamer::BitmapProviderPrx gBitmapProvider;

// General Settings
const long TARGET_FPS = 200;
const long REFRESH_DELAY = 1000/TARGET_FPS; //ms

// GPU
int gBlockSize(1024);
int gSharedMemSize(64);

// Rendering window vars
unsigned int gWindowWidth  = 512;
unsigned int gWindowHeight = 512;
unsigned int gWindowDepth  = 4;

// ----------------------------------------------------------------------
// Scene
// ----------------------------------------------------------------------
float4 gBkColor = {1.f, 1.f, 1.f, 0.f};
int   gTotalPathTracingIterations = 1;

::IceStreamer::SceneInfo gSceneInfo = 
{ 
   gWindowWidth,               // width
   gWindowHeight,              // height
   true,                       // shadowsEnabled
   5,                          // nbRayIterations
   3.f,                        // transparentColor
   20000.f,                    // viewDistance
   0.9f,                       // shadowIntensity
   20.f,                       // width3DVision
   gBkColor.x,                 // backgroundColor
   gBkColor.y,                 // backgroundColor
   gBkColor.z,                 // backgroundColor
   gBkColor.w,                 // backgroundColor
   0,                          // supportFor3DVision
   0,                          // renderBoxes
   0,                          // pathTracingIteration
   gTotalPathTracingIterations,// maxPathTracingIterations
   ::IceStreamer::otOpenGL,    // outputType
   0,                          // Timer
   1,                          // Fog
   1                           // Isometric3D
};

bool gRefreshNeeded(true);

bool gAnimate(false);
long gTickCount(0);

// Camera
float4 gViewPos;
float4 gViewDir;
float4 gViewAngles;
float4 gPreviewViewPos = {0.f, 0.f, 0.f, 0.f};
float4 gLampPosition;
float4 gLampDirection;

#ifdef USE_KINECT
int gSkeletonPrimitiveIndex    = -1;
int gSkeletonBoxID             = 2;
const float gSkeletonSize      = 500.f;
const float gSkeletonThickness =  60.f;
float4 gSkeletonPosition       = { 0.f, 0.f, 0.f, 0.f };
#endif // USE_KINECT

// Post processing
::IceStreamer::PostProcessingInfo gPostProcessingInfo = { 0, 4000.f, 40.f, 100 };

// --------------------------------------------------------------------------------
// OpenGL
// --------------------------------------------------------------------------------
GLubyte* gUbImage;
int gTimebase(0);
int gFrame(0);
int gFPS(0);
bool gHelp(false);

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
bool bNoPrompt = false;  

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;

/*
________________________________________________________________________________

vectorRotation
________________________________________________________________________________
*/
void vectorRotation( float4& vector, float4 angles )
{ 
	float4 result = vector;
	/* X axis */
	result.y = vector.y*cos(angles.x) - vector.z*sin(angles.x);
	result.z = vector.y*sin(angles.x) + vector.z*cos(angles.x);
	vector = result;
	result = vector;
	/* Y axis */
	result.z = vector.z*cos(angles.y) - vector.x*sin(angles.y);
	result.x = vector.z*sin(angles.y) + vector.x*cos(angles.y);
	vector = result;
}

/*
________________________________________________________________________________

getRandomValue
________________________________________________________________________________
*/
float getRandomValue( int range, int safeZone, bool allowNegativeValues = true )
{
	float value( static_cast<float>(rand()%range) + safeZone);
	if( allowNegativeValues ) 
	{
		value *= (rand()%2==0)? -1 : 1;
	}
	return value;
}

/*
________________________________________________________________________________

idle
________________________________________________________________________________
*/
void idle()
{
}

/*
________________________________________________________________________________

cleanup
________________________________________________________________________________
*/
void cleanup()
{
}

/*
--------------------------------------------------------------------------------
the window and assign callbacks
--------------------------------------------------------------------------------
*/
void initgl( int argc, char **argv )
{
	size_t len(gWindowWidth*gWindowHeight*gWindowDepth);
	gUbImage = new GLubyte[len];

	glutInit(&argc, (char**)argv);
	glutInitDisplayMode( GLUT_RGB | GLUT_DOUBLE );
	glutInitWindowPosition(glutGet(GLUT_SCREEN_WIDTH)/2 - gWindowWidth/2, glutGet(GLUT_SCREEN_HEIGHT)/2 - gWindowHeight/2);

	glutInitWindowSize(gWindowWidth, gWindowHeight);
	glutCreateWindow("Ray-tracer Client");

	glutDisplayFunc(display);       // register GLUT callback functions
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutTimerFunc(REFRESH_DELAY,timerEvent,1);

	return;
}

void RenderString(float x, float y, void *font, const std::string& string, const float4& rgb)
{
  glColor3f(rgb.x, rgb.y, rgb.z);
  glRasterPos2f(x, y);

  glutBitmapString( font, reinterpret_cast<const unsigned char*>(string.c_str()) );
}

void TexFunc(void)
{
	glEnable(GL_TEXTURE_2D);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);

	glTexImage2D(GL_TEXTURE_2D, 0, 3, gWindowWidth, gWindowHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, gUbImage);

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

   float4 textColor = {1.f, 1.f, 1.f, 0.f};
   if( gHelp )
   {
      gFrame++;
      int time=glutGet(GLUT_ELAPSED_TIME);
      if( time - gTimebase > 1000 )
      {
         gFPS = gFrame*1000/(time-gTimebase);
         gTimebase = time;
         gFrame=0;
      }

      char tmp[1024];
      strcpy(tmp, "b: Randomly change background color\n");
      strcat(tmp, "B: Reset background color to black\n");
      strcat(tmp, "d: Enable/Disable depth of field post processing effect\n");
      strcat(tmp, "i: Switch Boxes/Primitives\n");
      strcat(tmp, "m: Automatic animation for performance testing\n");
      strcat(tmp, "n: Next protein\n");
      strcat(tmp, "o: Increase number of blocks\n");
      strcat(tmp, "p: Increase shared memory\n");
      strcat(tmp, "s: Enable/Disable shadows\n");
      strcat(tmp, "1: Decrease depth of field post processing effect\n");
      strcat(tmp, "2: Increase depth of field post processing effect\n");
      strcat(tmp, "4: Decrease view distance\n");
      strcat(tmp, "5: Increase view distance\n");
      strcat(tmp, "7: Decrease 3DVision distance between eyes\n");
      strcat(tmp, "8: Increase 3DVision distance between eyes\n");
      strcat(tmp, "9: Increase shadow intensity\n");
      strcat(tmp, "-: Decrease number of ray iterations\n");
      strcat(tmp, "+: Increase number of ray iterations\n");
      strcat(tmp, "h: Help\n");
      strcat(tmp, "Escape: Exit application\n");
      RenderString(-0.9f, 0.9f, GLUT_BITMAP_HELVETICA_10, tmp, textColor );
   }
   RenderString(-0.9f, -0.9f, GLUT_BITMAP_HELVETICA_10, "Copyright(C) Cyrille Favreau - http://cudaopencl.blogspot.com", textColor );

	glFlush();

	glutSwapBuffers();
}

void timerEvent(int value)
{
   if( gRefreshNeeded )
   {
      try 
      {
         IceStreamer::bytes bitmap = gBitmapProvider->getBitmap( 
            gViewPos.x, gViewPos.y, gViewPos.z, 
            gViewDir.x, gViewDir.y, gViewDir.z, 
            gViewAngles.x, gViewAngles.y, gViewAngles.z,
            gSceneInfo, gPostProcessingInfo );
         
         for( unsigned int i(0); i<bitmap.size(); ++i) 
            gUbImage[i] = bitmap[i];
      }
      catch(const Ice::Exception& e)
      {
         std::cout << e.ice_name() << std::endl;
         std::cout << e.ice_file() << std::endl;
         std::cout << e.ice_stackTrace() << std::endl;
      }
      catch( ... ) 
      {
         std::cout << "Unknown exception" << std::endl;
      }

#ifdef WIN32
      long t = GetTickCount();
      anim += (t - gTickCount) / 1000.f;
      gTickCount = t;
#else
      anim += 0.2f;
#endif
   }
   glutPostRedisplay();
   glutTimerFunc(REFRESH_DELAY, timerEvent,0);
   gRefreshNeeded = false;
}

// Keyboard events handler
//*****************************************************************************
void keyboard(unsigned char key, int x, int y)
{
	srand(static_cast< unsigned int>(time(NULL))); 

	switch(key) 
	{
	case 'f':
		{
			// Toggle to full screen mode
			glutFullScreen();
			break;
		}
   case '+':
      {
         gSceneInfo.nbRayIterations++;
         gSceneInfo.nbRayIterations = (gSceneInfo.nbRayIterations>10) ? 10 : gSceneInfo.nbRayIterations;
         break;
      }
   case '-':
      {
         gSceneInfo.nbRayIterations--;
         gSceneInfo.nbRayIterations = (gSceneInfo.nbRayIterations<1) ? 1 : gSceneInfo.nbRayIterations;
         break;
      }
	case 'm':
		{
         gAnimate = !gAnimate;
			break;
		}
   case 'b':
      {
         gSceneInfo.backgroundColorR = static_cast<float>(rand()%100/100.f);
         gSceneInfo.backgroundColorG = static_cast<float>(rand()%100/100.f);
         gSceneInfo.backgroundColorB = static_cast<float>(rand()%100/100.f);
         break;
      }
   case 'B':
      {
         gSceneInfo.backgroundColorR = 0.f;
         gSceneInfo.backgroundColorG = 0.f;
         gSceneInfo.backgroundColorB = 0.f;
         break;
      }
   case 'i':
      {
         gSceneInfo.renderBoxes = !gSceneInfo.renderBoxes;
         break;
      }
   case 'h':
      {
         gHelp = !gHelp;
         break;
      }
   case 's':
      {
         gSceneInfo.shadowsEnabled = !gSceneInfo.shadowsEnabled;
         break;
      }
   case '3':
      {
         gSceneInfo.supportFor3DVision = !gSceneInfo.supportFor3DVision;
         break;
      }
   case '4':
      {
         gSceneInfo.viewDistance -= 200.f;
         break;
      }
   case '5':
      {
         gSceneInfo.viewDistance += 200.f;
         break;
      }
   case '7':
      {
         gSceneInfo.width3DVision += 10.f;
         break;
      }
   case '8':
      {
         gSceneInfo.width3DVision -= 10.f;
         break;
      }
   case '9':
      {
         gSceneInfo.shadowIntensity += 0.1f;
         gSceneInfo.shadowIntensity = (gSceneInfo.shadowIntensity > 1.f) ? 0.f : gSceneInfo.shadowIntensity;
         break;
      }
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
   gRefreshNeeded = true;
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
	gViewAngles.x = 0.f;
	gViewAngles.y = 0.f;
}

void motion(int x, int y)
{
	switch( mouse_buttons ) 
	{
	case 1:
      // Translate
      if( glutGetModifiers() != GLUT_ACTIVE_CTRL ) 
      {
         // Moving back and forth
         gViewPos.z += 10*(mouse_old_y-y);
         if( glutGetModifiers() != GLUT_ACTIVE_SHIFT ) 
         {
            // Changing camera angle
            gViewDir.z += 10*(mouse_old_y-y);
         }
      }
      else
      {
         // Depth of field focus
         gPostProcessingInfo.param1 += 4*(mouse_old_y-y);
      }
      gRefreshNeeded = true;
		break;
	case 4:
	{
		// Rotates the scene around X and Y axis
		gViewAngles.y = -asin( (mouse_old_x-x) / 100.f );
		gViewAngles.x =  asin( (mouse_old_y-y) / 100.f );
      gRefreshNeeded = true;
      break;
	}
	case 2:
		// Move gViewPos postion along X and Y axis
		gViewPos.x += (mouse_old_x-x);
		gViewPos.y += (mouse_old_y-y);
		gViewDir.x += (mouse_old_x-x);
		gViewDir.y += (mouse_old_y-y);
      gRefreshNeeded = true;
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
	if( gUbImage ) delete [] gUbImage;

	exit (iExitCode);
}

int main( int argc, char* argv[] )
{
	if( argc == 3 )
	{
		gWindowWidth  = atoi(argv[1]);
		gWindowHeight = atoi(argv[2]);
		gSceneInfo.width = gWindowWidth;
		gSceneInfo.height = gWindowHeight;
	}

   try 
   {
      gCommunicator = Ice::initialize(argc, argv);
      std::cout << "Connecting to Ice server ..." << std::endl;

      gBitmapProvider = ::IceStreamer::BitmapProviderPrx::checkedCast(
         gCommunicator->propertyToProxy("IceStreamerAdaptor.Proxy"));

      gSceneInfo = gBitmapProvider->getSceneInfo();
      gWindowWidth  = gSceneInfo.width;
      gWindowHeight = gSceneInfo.height;

      // Camera information
      gViewPos.x =     0.f;
      gViewPos.y =     0.f;
      gViewPos.z = -5000.f;

      gViewDir.x = 0.f;
      gViewDir.y = 0.f;
      gViewDir.z = 0.f;

      gViewAngles.x = 0.f;
      gViewAngles.y = 0.f;
      gViewAngles.z = 0.f;

      // First initialize OpenGL context, so we can properly set the GL for CUDA.
      // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
      initgl( argc, argv );

      atexit(cleanup);
      glutMainLoop();
   }
   catch(const Ice::Exception& e)
   {
      std::cout << e.ice_stackTrace() << std::endl;
   }
   catch( ... ) 
   {
      std::cout << "Unknown exception" << std::endl;
   }

   // Normally unused return path
   Cleanup(EXIT_SUCCESS);

	return 0;
}

