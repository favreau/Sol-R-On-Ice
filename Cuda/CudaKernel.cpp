/* 
* Cuda Raytracer
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

#include <math.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <sstream>

#include <vector_functions.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#ifdef LOGGING
#include <ETWLoggingModule.h>
#include <ETWResources.h>
#else
#define LOG_INFO( msg ) std::cout << msg << std::endl;
#define LOG_ERROR( msg ) std::cerr << msg << std::endl;
#endif

#include "CudaRayTracer.h"
#include "CudaKernel.h"

const long MAX_SOURCE_SIZE = 65535;
const long MAX_DEVICES = 10;

/*
* CudaKernel constructor
*/
CudaKernel::CudaKernel( int draft ) :
	m_hLevels(nullptr), m_levelsSize(0),
	m_hPrimitives(nullptr), m_nbActivePrimitives(-1), 
	m_hMaterials(nullptr), m_nbActiveMaterials(-1),
	m_hTextures(nullptr), m_nbActiveTextures(0),
	m_hLamps(nullptr), m_nbActiveLamps(-1),
	m_hRandoms(nullptr),
	m_hDepthOfField(nullptr),
#if USE_KINECT
	m_hVideo(nullptr), m_hDepth(nullptr),
	m_skeletons(0), m_hNextDepthFrameEvent(0), m_hNextVideoFrameEvent(0), m_hNextSkeletonEvent(0),
	m_pVideoStreamHandle(0), m_pDepthStreamHandle(0),
	m_skeletonsBody(-1), m_skeletonsLamp(-1), m_skeletonIndex(-1),
#endif // USE_KINECT
	m_initialDraft(draft), m_draft(1),
	m_texturedTransfered(false),
   m_blockSize(16,16,0)
{
#ifdef LOGGING
	// Initialize Log
	LOG_INITIALIZE_ETW(
		&GPU_CudaRAYTRACERMODULE,
		&GPU_CudaRAYTRACERMODULE_EVENT_DEBUG,
		&GPU_CudaRAYTRACERMODULE_EVENT_VERBOSE,
		&GPU_CudaRAYTRACERMODULE_EVENT_INFO, 
		&GPU_CudaRAYTRACERMODULE_EVENT_WARNING,
		&GPU_CudaRAYTRACERMODULE_EVENT_ERROR);
#endif // NDEBUG

#if USE_KINECT
	// Initialize Kinect
	NuiInitialize( NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX | NUI_INITIALIZE_FLAG_USES_SKELETON | NUI_INITIALIZE_FLAG_USES_COLOR);

	m_hNextDepthFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL ); 
	m_hNextVideoFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL ); 
	m_hNextSkeletonEvent   = CreateEvent( NULL, TRUE, FALSE, NULL );

	m_skeletons = CreateEvent( NULL, TRUE, FALSE, NULL );			 
	NuiSkeletonTrackingEnable( m_skeletons, 0 );

	NuiImageStreamOpen( NUI_IMAGE_TYPE_COLOR,                  NUI_IMAGE_RESOLUTION_640x480, 0, 2, m_hNextVideoFrameEvent, &m_pVideoStreamHandle );
	NuiImageStreamOpen( NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX, NUI_IMAGE_RESOLUTION_320x240, 0, 2, m_hNextDepthFrameEvent, &m_pDepthStreamHandle );

	NuiCameraElevationSetAngle( 0 );
#endif // USE_KINECT

	// Eye position
	m_viewPos.x =   0.0f;
	m_viewPos.y =   0.0f;
	m_viewPos.z = -40.0f;

	// Rotation angles
	m_angles.x = 0.0f;
	m_angles.y = 0.0f;
	m_angles.z = 0.0f;
}

void CudaKernel::initializeDevice(
	int        width, 
	int        height, 
	int        nbPrimitives,
	int        nbLamps,
	int        nbMaterials,
	int        nbTextures,
	int*       levels,
	int        levelsSize)
{
   m_imageWidth  = width;
   m_imageHeight = height;

	m_hLevels = levels;
	m_levelsSize = levelsSize;

	initialize_scene( m_imageWidth, m_imageHeight, nbPrimitives, nbLamps, nbMaterials, nbTextures, levelsSize );

	// Setup device memory
	m_hPrimitives = new Primitive[nbPrimitives];
	memset( m_hPrimitives, 0, nbPrimitives*sizeof(Primitive) ); 
	m_hLamps = new Lamp[nbLamps];
	memset( m_hLamps, 0, nbLamps*sizeof(Lamp) ); 
	m_hMaterials = new Material[nbMaterials];
	memset( m_hMaterials, 0, nbMaterials*sizeof(Material) ); 
	m_hTextures = new char[gTextureWidth*gTextureHeight*gColorDepth*nbTextures];

	// Randoms
	m_hRandoms = new float[width*height];
	int i;
#pragma omp parallel for
	for( i=0; i<width*height; ++i)
	{
		m_hRandoms[i] = (rand()%1000-500)/500.f;
	}
}

void CudaKernel::releaseDevice()
{
	LOG_INFO("Release device memory\n");
	finalize_scene();

	delete m_hPrimitives;
	delete m_hLamps;
	delete m_hMaterials;
	delete m_hTextures;
	delete m_hRandoms;
}

/*
* runKernel
*/
void CudaKernel::render( 
   float4 eye, float4 dir, float4 angles,
   unsigned char* bitmap,
	float timer,
	float pointOfFocus,
	float transparentColor)
{
   // Camera
   m_viewPos   = eye;
   m_viewDir   = dir;
   m_angles.x  += angles.x;
   m_angles.y  += angles.y;
   m_angles.z  += angles.z;
   m_draft     = m_initialDraft;

#if USE_KINECT
	// Video
	const NUI_IMAGE_FRAME* pImageFrame = 0;
	WaitForSingleObject (m_hNextVideoFrameEvent,INFINITE); 
	HRESULT status = NuiImageStreamGetNextFrame( m_pVideoStreamHandle, 0, &pImageFrame ); 
	if(( status == S_OK) && pImageFrame ) 
	{
		INuiFrameTexture* pTexture = pImageFrame->pFrameTexture;
		NUI_LOCKED_RECT LockedRect;
		pTexture->LockRect( 0, &LockedRect, NULL, 0 ) ; 
		if( LockedRect.Pitch != 0 ) 
		{
			m_hVideo = (char*) LockedRect.pBits;
		}
	}

	// Depth
	const NUI_IMAGE_FRAME* pDepthFrame = 0;
	WaitForSingleObject (m_hNextDepthFrameEvent,INFINITE); 
	status = NuiImageStreamGetNextFrame( m_pDepthStreamHandle, 0, &pDepthFrame ); 
	if(( status == S_OK) && pDepthFrame ) 
	{
		INuiFrameTexture* pTexture = pDepthFrame->pFrameTexture;
		if( pTexture ) 
		{
			NUI_LOCKED_RECT LockedRectDepth;
			pTexture->LockRect( 0, &LockedRectDepth, NULL, 0 ) ; 
			if( LockedRectDepth.Pitch != 0 ) 
			{
				m_hDepth = (char*) LockedRectDepth.pBits;
			}
		}
	}
	NuiImageStreamReleaseFrame( m_pVideoStreamHandle, pImageFrame ); 
	NuiImageStreamReleaseFrame( m_pDepthStreamHandle, pDepthFrame ); 

   // copy kinect data to GPU
   h2d_kinect( 
      m_hVideo, gKinectVideo*gKinectVideoHeight*gKinectVideoWidth,
      m_hDepth, gKinectDepth*gKinectDepthHeight*gKinectDepthWidth );
#endif // USE_KINECT

	// Initialise Input arrays
	h2d_scene( m_hPrimitives, m_nbActivePrimitives+1, m_hLamps, m_nbActiveLamps+1 );

	if( !m_texturedTransfered )
	{
		h2d_materials( 
         m_hMaterials, m_nbActiveMaterials+1, 
         m_hTextures,  m_nbActiveTextures, 
         m_hRandoms,   m_imageWidth*m_imageHeight,
         m_hLevels,    m_levelsSize);
		m_texturedTransfered = true;
	}

#ifdef USE_KINECT
#endif // KINECT

	// Run the ray-tracing kernel!!
	cudaRender(
      m_blockSize,
		m_nbActivePrimitives+1, m_nbActiveLamps+1,
		eye, dir, angles, 
		m_imageWidth, m_imageHeight,
		pointOfFocus, m_draft, transparentColor, timer );
	d2h_bitmap( bitmap, m_imageWidth*m_imageHeight*gColorDepth);

	m_draft--;
	m_draft = (m_draft < 1) ? 1 : m_draft;
}

/*
*
*/
CudaKernel::~CudaKernel()
{
	// Clean up
	releaseDevice();

#if USE_KINECT
	CloseHandle(m_skeletons);
	CloseHandle(m_hNextDepthFrameEvent); 
	CloseHandle(m_hNextVideoFrameEvent); 
	CloseHandle(m_hNextSkeletonEvent);
	NuiShutdown();
#endif // USE_KINECT
}

float4 CudaKernel::normalVector( float4 v1, float4 v2 )
{
	float4 result;
	result.x = v1.y*v2.z - v1.z*v2.y;
	result.y = v1.z*v2.x - v1.x*v2.z;
	result.z = v1.x*v2.y - v1.y*v2.x;
	result.w = 0.f;
	return result;
}

float CudaKernel::dotProduct( float4 v1, float4 v2 )
{
	return ( v1.x*v2.x + v1.y*v2.y + v1.z*v2.z);
}

float CudaKernel::vectorLength( const float4 v )
{
	return sqrt( v.x*v.x + v.y*v.y + v.z*v.z );
}

void CudaKernel::normalizeVector( float4 &v )
{
	float d = vectorLength( v );
	v.x /= d;
	v.y /= d;
	v.z /= d;
}

long CudaKernel::addPrimitive( PrimitiveType type )
{
   m_nbActivePrimitives++;
	m_hPrimitives[m_nbActivePrimitives].type = type;
	m_hPrimitives[m_nbActivePrimitives].materialId = NO_MATERIAL;
	return m_nbActivePrimitives;
}

void CudaKernel::setPrimitive( 
	int index, 
	float x0, float y0, float z0, 
	float width, 
	float height, 
	int   materialId, 
	int   materialPaddingX, int materialPaddingY )
{
	setPrimitive( index, x0, y0, z0, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, width, height, materialId, materialPaddingX, materialPaddingY );
}

void CudaKernel::setPrimitive( 
	int index, 
	float x0, float y0, float z0, 
	float x1, float y1, float z1, 
	float x2, float y2, float z2, 
	float width, 
	float height, 
	int   materialId, 
	int   materialPaddingX, int materialPaddingY )
{
	if( index>=0 && index<=m_nbActivePrimitives) 
	{
		m_hPrimitives[index].p0.x   = x0;
		m_hPrimitives[index].p0.y   = y0;
		m_hPrimitives[index].p0.z   = z0;
		m_hPrimitives[index].p0.w   = width;

		switch( m_hPrimitives[index].type )
		{
		case ptTriangle:
			{
				m_hPrimitives[index].p0.x   = x1-x0;
				m_hPrimitives[index].p0.y   = y1-y0;
				m_hPrimitives[index].p0.z   = z1-z0;

				m_hPrimitives[index].p1.x   = x2-x1;
				m_hPrimitives[index].p1.y   = y2-y1;
				m_hPrimitives[index].p1.z   = z2-z1;

				m_hPrimitives[index].p2.x   = x0-x2;
				m_hPrimitives[index].p2.y   = y0-y2;
				m_hPrimitives[index].p2.z   = z0-z2;

				m_hPrimitives[index].normal = normalVector(m_hPrimitives[index].p0, m_hPrimitives[index].p1);
				m_hPrimitives[index].v0     = normalVector(m_hPrimitives[index].p0, m_hPrimitives[index].normal);
				m_hPrimitives[index].v1     = normalVector(m_hPrimitives[index].p1, m_hPrimitives[index].normal);
				m_hPrimitives[index].v2     = normalVector(m_hPrimitives[index].p2, m_hPrimitives[index].normal);

				float4 a;
				a.x = x0;
				a.y = y0;
				a.z = z0;
				a.w = 0.f;
				m_hPrimitives[index].normal.w = dotProduct(a,m_hPrimitives[index].normal);
				break;
			}
		case ptXYPlane:
			{
				m_hPrimitives[index].normal.x = 0.f;
				m_hPrimitives[index].normal.y = 0.f;
				m_hPrimitives[index].normal.z = 1.f;
				break;
			}
		case ptYZPlane:
			{
				m_hPrimitives[index].normal.x = 1.f;
				m_hPrimitives[index].normal.y = 0.f;
				m_hPrimitives[index].normal.z = 0.f;
				break;
			}
		case ptXZPlane:
		case ptCheckboard:
			{
				m_hPrimitives[index].normal.x = 0.f;
				m_hPrimitives[index].normal.y = 1.f;
				m_hPrimitives[index].normal.z = 0.f;
				break;
			}
		default:
			{
				break;
			}
		}

		m_hPrimitives[index].size.x = width;
		m_hPrimitives[index].size.y = height;
		m_hPrimitives[index].size.z = 0.f;
		m_hPrimitives[index].size.w = 0.f; // Not used
		m_hPrimitives[index].materialId    = materialId;
		m_hPrimitives[index].materialInfo.x = (gTextureWidth /width /2)*materialPaddingX;
		m_hPrimitives[index].materialInfo.y = (gTextureHeight/height/2)*materialPaddingY;
	}
}

void CudaKernel::rotatePrimitive( 
	int   index, 
	float x, 
	float y,
	float z )
{
	if( index>=0 && index<=m_nbActivePrimitives) 
	{
		float4 v = m_hPrimitives[index].normal;
		float4 r = v;
		/* X axis */
		r.y = v.y*cos(x) - v.z*sin(x);
		r.z = v.y*sin(x) + v.z*cos(x);
		v = r;
		r = v;
		/* Y axis */ \
			r.z = v.z*cos(y) - v.x*sin(y);
		r.x = v.z*sin(y) + v.x*cos(y);
		m_hPrimitives[index].normal = r;
	}
}

void CudaKernel::translatePrimitive( 
	int   index, 
	float x, 
	float y, 
	float z )
{
	if( index>=0 && index<=m_nbActivePrimitives) 
	{
		setPrimitive( 
			index,
			m_hPrimitives[index].p0.x+x, m_hPrimitives[index].p0.y+y, m_hPrimitives[index].p0.z+z,
			m_hPrimitives[index].p1.x  , m_hPrimitives[index].p1.y  , m_hPrimitives[index].p1.z,
			m_hPrimitives[index].p2.x  , m_hPrimitives[index].p2.y  , m_hPrimitives[index].p2.z,
			m_hPrimitives[index].size.x,m_hPrimitives[index].size.y,
			m_hPrimitives[index].materialId, 1, 1 );


	}
}

void CudaKernel::getPrimitiveCenter(
	int   index, 
	float& x, 
	float& y, 
	float& z,
	float& w)
{
	if( index>=0 && index<=m_nbActivePrimitives) 
	{
		x = m_hPrimitives[index].p0.x;
		y = m_hPrimitives[index].p0.y;
		z = m_hPrimitives[index].p0.z;
		w = m_hPrimitives[index].p0.w;
	}
}

void CudaKernel::setPrimitiveCenter(
	int   index, 
	float x, 
	float y, 
	float z,
	float w)
{
	if( index>=0 && index<=m_nbActivePrimitives) 
	{
		m_hPrimitives[index].p0.x = x;
		m_hPrimitives[index].p0.y = y;
		m_hPrimitives[index].p0.z = z;
		m_hPrimitives[index].p0.w = w;

		m_hPrimitives[index].size.x = w;
	}
}

long CudaKernel::addCube( 
	float x, float y, float z, 
	float radius, 
	int   martialId, 
	int   materialPaddingX, int materialPaddingY )
{
	return addRectangle(x,y,z,radius,radius,radius,martialId,materialPaddingX,materialPaddingY);
}

long CudaKernel::addRectangle( 
	float x, float y, float z, 
	float width, float height,
	float depth,
	int   martialId, 
	int   materialPaddingX, int materialPaddingY )
{
	long returnValue;
	// Back
	returnValue = addPrimitive( ptXYPlane );
	setPrimitive( returnValue, x, y, z+depth, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, width, height, martialId, materialPaddingX, materialPaddingY ); 

	// Front
	returnValue = addPrimitive( ptXYPlane );
	setPrimitive( returnValue, x, y, z-depth, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, width, height, martialId, materialPaddingX, materialPaddingY ); 

	// Left
	returnValue = addPrimitive( ptYZPlane );
	setPrimitive( returnValue, x-width, y, z, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, depth, height, martialId, materialPaddingX, materialPaddingY ); 

	// Right
	returnValue = addPrimitive( ptYZPlane );
	setPrimitive( returnValue, x+width, y, z, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, depth, height, martialId, materialPaddingX, materialPaddingY ); 

	// Top
	returnValue = addPrimitive( ptXZPlane );
	setPrimitive( returnValue, x, y+height, z, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, width, depth, martialId, materialPaddingX, materialPaddingY ); 

	// Bottom
	returnValue = addPrimitive( ptXZPlane );
	setPrimitive( returnValue, x, y-height, z, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, width, depth, martialId, materialPaddingX, materialPaddingY ); 
	return returnValue;
}

void CudaKernel::setPrimitiveMaterial( 
	int   index, 
	int   materialId,
	int   materialOffsetX,
	int   materialOffsetY)
{
	if( index>=0 && index<=m_nbActivePrimitives) {
		m_hPrimitives[index].materialId = materialId;
	}
}

long CudaKernel::addLamp( LampType lampType )
{
	m_nbActiveLamps++;
   m_hLamps[m_nbActiveLamps].lampType = lampType;
	return m_nbActiveLamps;
}

void CudaKernel::setLamp( 
   int index,
   float x, float y, float z, 
   float width, float height,
   float r, float g, float b, float intensity )
{
	if( index>=0 && index<=m_nbActiveLamps ) 
   {
		m_hLamps[index].center.x   = x;
		m_hLamps[index].center.y   = y;
		m_hLamps[index].center.z   = z;
		m_hLamps[index].center.w   = width;
		m_hLamps[index].color.x    = r;
		m_hLamps[index].color.y    = g;
		m_hLamps[index].color.z    = b;
		m_hLamps[index].intensity  = intensity;
	}
}

// ---------- Materials ----------
long CudaKernel::addMaterial()
{
   m_nbActiveMaterials++;
	return m_nbActiveMaterials;
}

void CudaKernel::setMaterial( 
	int   index,
	float r, float g, float b, 
	float reflection, 
	float refraction, 
	int   textured,
	float transparency,
	int   textureId,
	float specValue, float specPower, float specCoef, float innerIllumination )
{
	if( index>=0 && index<=m_nbActiveMaterials ) 
   {
		m_hMaterials[index].color.x     = r;
		m_hMaterials[index].color.y     = g;
		m_hMaterials[index].color.z     = b;
      m_hMaterials[index].color.w     = 0.f;
      m_hMaterials[index].specular.x  = specValue;
      m_hMaterials[index].specular.y  = specPower;
      m_hMaterials[index].specular.z  = innerIllumination;
      m_hMaterials[index].specular.w  = specCoef;
		m_hMaterials[index].reflection  = reflection;
		m_hMaterials[index].refraction  = refraction;
      m_hMaterials[index].transparency= transparency;
		m_hMaterials[index].texture.x   = textured;
		m_hMaterials[index].texture.y   = textureId;
	}
}

// ---------- Textures ----------
void CudaKernel::setTexture(
	int   index,
	char* texture )
{
	char* idx = m_hTextures+index*gTextureWidth*gTextureHeight*gTextureDepth;
	int j(0);
	for( int i(0); i<gTextureWidth*gTextureHeight*gColorDepth; i += gColorDepth ) {
		idx[j]   = texture[i+2];
		idx[j+1] = texture[i+1];
		idx[j+2] = texture[i];
		j+=gTextureDepth;
	}
}

/*
*
*/
char* CudaKernel::loadFromFile( const std::string& filename, size_t& length )
{
	// Load the kernel source code into the array source_str
	FILE *fp = 0;
	char *source_str = 0;

	fopen_s( &fp, filename.c_str(), "r");
	if( fp == 0 ) 
	{
		std::cout << "Failed to load kernel " << filename.c_str() << std::endl;
	}
	else 
	{
		source_str = (char*)malloc(MAX_SOURCE_SIZE);
		length = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
		fclose( fp );
	}
	return source_str;
}

// ---------- Kinect ----------
long CudaKernel::addTexture( const std::string& filename )
{
	FILE *filePtr(0); //our file pointer
	BITMAPFILEHEADER bitmapFileHeader; //our bitmap file header
	char *bitmapImage;  //store image data
	BITMAPINFOHEADER bitmapInfoHeader;
	DWORD imageIdx=0;  //image index counter
	char tempRGB;  //our swap variable

	//open filename in read binary mode
	fopen_s(&filePtr, filename.c_str(), "rb");
	if (filePtr == NULL) {
		return 1;
	}

	//read the bitmap file header
	fread(&bitmapFileHeader, sizeof(BITMAPFILEHEADER), 1, filePtr);

	//verify that this is a bmp file by check bitmap id
	if (bitmapFileHeader.bfType !=0x4D42) {
		fclose(filePtr);
		return 1;
	}

	//read the bitmap info header
	fread(&bitmapInfoHeader, sizeof(BITMAPINFOHEADER),1,filePtr);

	//move file point to the begging of bitmap data
	fseek(filePtr, bitmapFileHeader.bfOffBits, SEEK_SET);

	//allocate enough memory for the bitmap image data
	bitmapImage = (char*)malloc(bitmapInfoHeader.biSizeImage);

	//verify memory allocation
	if (!bitmapImage)
	{
		free(bitmapImage);
		fclose(filePtr);
		return 1;
	}

	//read in the bitmap image data
	fread( bitmapImage, bitmapInfoHeader.biSizeImage, 1, filePtr);

	//make sure bitmap image data was read
	if (bitmapImage == NULL)
	{
		fclose(filePtr);
		return NULL;
	}

	//swap the r and b values to get RGB (bitmap is BGR)
	for (imageIdx = 0; imageIdx < bitmapInfoHeader.biSizeImage; imageIdx += 3)
	{
		tempRGB = bitmapImage[imageIdx];
		bitmapImage[imageIdx] = bitmapImage[imageIdx + 2];
		bitmapImage[imageIdx + 2] = tempRGB;
	}

	//close file and return bitmap image data
	fclose(filePtr);

	char* index = m_hTextures + (m_nbActiveTextures*bitmapInfoHeader.biSizeImage);
	memcpy( index, bitmapImage, bitmapInfoHeader.biSizeImage );
	m_nbActiveTextures++;

	free( bitmapImage );
	return m_nbActiveTextures-1;
}

#ifdef USE_KINECT
long CudaKernel::updateSkeletons( 
	double center_x, double  center_y, double  center_z, 
	double size,
	double radius,       int materialId,
	double head_radius,  int head_materialId,
	double hands_radius, int hands_materialId,
	double feet_radius,  int feet_materialId)
{
	bool found = false;
	HRESULT hr = NuiSkeletonGetNextFrame( 0, &m_skeletonFrame );
	if( hr == S_OK )
	{
		int i=0;
		while( i<NUI_SKELETON_COUNT && !found ) 
		{
			if( m_skeletonFrame.SkeletonData[i].eTrackingState == NUI_SKELETON_TRACKED ) 
			{
				found = true;
				if( m_skeletonIndex == -1 ) 
				{
					// Create Skeleton
					m_skeletonsBody = m_nbActivePrimitives+1;
					m_skeletonsLamp = m_nbActiveLamps+1;
					m_skeletonIndex = i;
					for( int j=0; j<20; j++ ) addPrimitive( ptSphere );
				}
				else 
				{
					for( int j=0; j<20; j++ ) 
					{
						double r = radius;
						int   m = materialId;
						bool createSphere(true);
						switch (j) {
						case NUI_SKELETON_POSITION_FOOT_LEFT:
						case NUI_SKELETON_POSITION_FOOT_RIGHT:
							r = feet_radius;
							m = feet_materialId;
							createSphere = true;
							break;
						case NUI_SKELETON_POSITION_HAND_LEFT:
						case NUI_SKELETON_POSITION_HAND_RIGHT:
							r = hands_radius;
							m = hands_materialId;
							createSphere = true;
							break;
						case NUI_SKELETON_POSITION_HEAD:
							r = head_radius;
							m = head_materialId;
							createSphere = true;
							break;
						}
						if( createSphere ) setPrimitive(
							m_skeletonsBody+j,
							static_cast<float>(     m_skeletonFrame.SkeletonData[i].SkeletonPositions[j].x * size + center_x),
							static_cast<float>(     m_skeletonFrame.SkeletonData[i].SkeletonPositions[j].y * size + center_y),
							static_cast<float>( center_z - 2.f*m_skeletonFrame.SkeletonData[i].SkeletonPositions[j].z * size ),
							static_cast<float>(r), 
							static_cast<float>(r), 
							m,
							1, 1 );
					}
				}
			}
			i++;
		}
	}
	return hr;
}

bool CudaKernel::getSkeletonPosition( int index, float4& position )
{
	bool returnValue(false);
	if( m_skeletonIndex != -1 ) 
	{
		position.x = m_skeletonFrame.SkeletonData[m_skeletonIndex].SkeletonPositions[index].x;
      position.y = m_skeletonFrame.SkeletonData[m_skeletonIndex].SkeletonPositions[index].y;
      position.z = m_skeletonFrame.SkeletonData[m_skeletonIndex].SkeletonPositions[index].z;
		returnValue = true;
	}
	return returnValue;
}

#endif // USE_KINECT

void CudaKernel::deviceQuery()
{
   std::cout << " CUDA Device Query (Runtime API) version (CUDART static linking)" << std::endl;

   int deviceCount = 0;
   cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

   if (error_id != cudaSuccess)
   {
      std::cout << "cudaGetDeviceCount returned " << (int)error_id << " -> " << cudaGetErrorString(error_id) << std::endl;
   }

   // This function call returns 0 if there are no CUDA capable devices.
   if (deviceCount == 0)
      std::cout << "There is no device supporting CUDA" << std::endl;
   else
      std::cout << "Found " << deviceCount << " CUDA Capable device(s)" << std::endl;

   int dev, driverVersion = 0, runtimeVersion = 0;

   for (dev = 0; dev < deviceCount; ++dev)
   {
      cudaSetDevice(dev);
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, dev);

      std::cout << "Device :" << dev <<", " << deviceProp.name << std::endl;

#if CUDART_VERSION >= 2020
      // Console log
      cudaDriverGetVersion(&driverVersion);
      cudaRuntimeGetVersion(&runtimeVersion);
      std::cout << "  CUDA Driver Version / Runtime Version          " << driverVersion/1000 << 
         "." << (driverVersion%100)/10 << 
         " / " << runtimeVersion/1000 << "." 
         << (runtimeVersion%100)/10 << std::endl;
#endif
      std::cout << "  CUDA Capability Major/Minor version number:    " << deviceProp.major << "." << deviceProp.minor << std::endl;

      std::cout << "  Total amount of global memory: " << 
         (float)deviceProp.totalGlobalMem/1048576.0f << "MBytes (" << 
         (unsigned long long) deviceProp.totalGlobalMem << " bytes)" << std::endl;

#if CUDART_VERSION >= 2000
      std::cout << "  (" << deviceProp.multiProcessorCount << ") Multiprocessors x (" << 
         _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) << ") CUDA Cores/MP:    " << 
         _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount << " CUDA Cores" << std::endl;
#endif
      std::cout << "  GPU Clock rate:                                " <<  deviceProp.clockRate * 1e-3f << "MHz (" << deviceProp.clockRate * 1e-6f << " GHz)" << std::endl;

#if CUDART_VERSION >= 4000

      /*
      // This is not available in the CUDA Runtime API, so we make the necessary calls the driver API to support this for output
      int memoryClock;
      getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
      std::cout << "  Memory Clock rate:                             " << memoryClock * 1e-3f << "Mhz" << std::endl;
      int memBusWidth;
      getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
      std::cout << "  Memory Bus Width:                              " << memBusWidth << "-bit" << std::endl;
      int L2CacheSize;
      getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

      if (L2CacheSize)
      {
         std::cout << "  L2 Cache Size:                                 " << L2CacheSize << "bytes" << std::endl;
      }
      */

      std::cout << "  Max Texture Dimension Size (x,y,z)             1D=(" << deviceProp.maxTexture1D << 
         "), 2D=(" << deviceProp.maxTexture2D[0] << "," << deviceProp.maxTexture2D[1] << 
         "), 3D=(" << deviceProp.maxTexture3D[0] << "," << deviceProp.maxTexture3D[1] << "," << deviceProp.maxTexture3D[2] << ")" << std::endl;
      std::cout << "  Max Layered Texture Size (dim) x layers        1D=(" << deviceProp.maxTexture1DLayered[0] <<
         ") x " << deviceProp.maxTexture1DLayered[1] 
         << ", 2D=(" << deviceProp.maxTexture2DLayered[0] << "," << deviceProp.maxTexture2DLayered[1] << 
         ") x " << deviceProp.maxTexture2DLayered[2] << std::endl;
#endif
      std::cout << "  Total amount of constant memory:               " << deviceProp.totalConstMem << "bytes" << std::endl;
      std::cout << "  Total amount of shared memory per block:       " << deviceProp.sharedMemPerBlock << "bytes" << std::endl;
      std::cout << "  Total number of registers available per block: " << deviceProp.regsPerBlock << std::endl;
      std::cout << "  Warp size:                                     " << deviceProp.warpSize << std::endl;
      std::cout << "  Maximum number of threads per multiprocessor:  " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
      std::cout << "  Maximum number of threads per block:           " << deviceProp.maxThreadsPerBlock << std::endl;
      std::cout << "  Maximum sizes of each dimension of a block:    " <<
         deviceProp.maxThreadsDim[0] << " x " <<
         deviceProp.maxThreadsDim[1] << " x " <<
         deviceProp.maxThreadsDim[2] << std::endl;

      m_blockSize.x = 32;
      m_blockSize.y = 32;
      m_blockSize.z = 1;

      std::cout << "  Maximum sizes of each dimension of a grid:     " <<
         deviceProp.maxGridSize[0] << " x " <<
         deviceProp.maxGridSize[1] << " x " <<
         deviceProp.maxGridSize[2] << std::endl;
      std::cout << "  Maximum memory pitch:                          " << deviceProp.memPitch << "bytes" << std::endl;
      std::cout << "  Texture alignment:                             " << deviceProp.textureAlignment  << "bytes" << std::endl;

#if CUDART_VERSION >= 4000
      std::cout << "  Concurrent copy and execution:                 " << (deviceProp.deviceOverlap ? "Yes" : "No") << " with " << deviceProp.asyncEngineCount << "copy engine(s)" << std::endl;
#else
      std::cout << "  Concurrent copy and execution:                 " << (deviceProp.deviceOverlap ? "Yes" : "No") << std::endl;
#endif

#if CUDART_VERSION >= 2020
      std::cout << "  Run time limit on kernels:                     " << (deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No") << std::endl;
      std::cout << "  Integrated GPU sharing Host Memory:            " << (deviceProp.integrated ? "Yes" : "No") << std::endl;
      std::cout << "  Support host page-locked memory mapping:       " << (deviceProp.canMapHostMemory ? "Yes" : "No") << std::endl;
#endif
#if CUDART_VERSION >= 3000
      std::cout << "  Concurrent kernel execution:                   " << (deviceProp.concurrentKernels ? "Yes" : "No") << std::endl;
      std::cout << "  Alignment requirement for Surfaces:            " << (deviceProp.surfaceAlignment ? "Yes" : "No") << std::endl;
#endif
#if CUDART_VERSION >= 3010
      std::cout << "  Device has ECC support enabled:                " << (deviceProp.ECCEnabled ? "Yes" : "No") << std::endl;
#endif
#if CUDART_VERSION >= 3020
      std::cout << "  Device is using TCC driver mode:               " << (deviceProp.tccDriver ? "Yes" : "No") << std::endl;
#endif
#if CUDART_VERSION >= 4000
      std::cout << "  Device supports Unified Addressing (UVA):      " << (deviceProp.unifiedAddressing ? "Yes" : "No") << std::endl;
      std::cout << "  Device PCI Bus ID / PCI location ID:           " << deviceProp.pciBusID << "/" << deviceProp.pciDeviceID << std::endl;
#endif

#if CUDART_VERSION >= 2020
      const char *sComputeMode[] =
      {
         "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
         "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
         "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
         "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
         "Unknown",
         NULL
      };
      std::cout << "  Compute Mode:" << std::endl;
      std::cout << "     < " << sComputeMode[deviceProp.computeMode] << " >" << std::endl;
#endif
   }

   // csv masterlog info
   // *****************************
   // exe and CUDA driver name
   std::cout << std::endl;
   std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
   char cTemp[10];

   // driver version
   sProfileString += ", CUDA Driver Version = ";
#ifdef WIN32
   sprintf_s(cTemp, 10, "%d.%d", driverVersion/1000, (driverVersion%100)/10);
#else
   sstd::cout << cTemp, "%d.%d", driverVersion/1000, (driverVersion%100)/10);
#endif
   sProfileString +=  cTemp;

   // Runtime version
   sProfileString += ", CUDA Runtime Version = ";
#ifdef WIN32
   sprintf_s(cTemp, 10, "%d.%d", runtimeVersion/1000, (runtimeVersion%100)/10);
#else
   sstd::cout << cTemp, "%d.%d", runtimeVersion/1000, (runtimeVersion%100)/10);
#endif
   sProfileString +=  cTemp;

   // Device count
   sProfileString += ", NumDevs = ";
#ifdef WIN32
   sprintf_s(cTemp, 10, "%d", deviceCount);
#else
   sstd::cout << cTemp, "%d", deviceCount);
#endif
   sProfileString += cTemp;

   // First 2 device names, if any
   for (dev = 0; dev < ((deviceCount > 2) ? 2 : deviceCount); ++dev)
   {
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, dev);
      sProfileString += ", Device = ";
      sProfileString += deviceProp.name;
   }

   sProfileString += "\n";
   std::cout << sProfileString.c_str() << std::endl;
}
