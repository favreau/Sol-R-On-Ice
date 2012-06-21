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

#pragma once

#include <stdio.h>
#include <string>
#include <windows.h>

#include "CudaDataTypes.h"
#if USE_KINECT
#include <nuiapi.h>
#endif // USE_KINECT

class CudaKernel
{
public:
	CudaKernel( int draft );
	~CudaKernel();

public:
	// ---------- Devices ----------
	void initializeDevice(
		int        width, 
		int        height, 
		int        nbPrimitives,
		int        nbLamps,
		int        nbMaterials,
		int        nbTextures,
		int*       levels,
		int        levelSize);
	void releaseDevice();

   void deviceQuery();

public:
	// ---------- Rendering ----------
	void render(
		unsigned char* bitmap,
		float timer,
		float pointOfFocus,
		float transparentColor );

public:

	// ---------- Primitives ----------
	long addPrimitive( PrimitiveType type );
	void setPrimitive( 
		int   index, 
		float x0, float y0, float z0, 
		float width, 
		float height, 
		int   martialId, 
		int   materialPaddingX, int materialPaddingY );
	void setPrimitive( 
		int   index, 
		float x0, float y0, float z0, 
		float x1, float y1, float z1, 
		float x2, float y2, float z2, 
		float width, 
		float height, 
		int   martialId, 
		int   materialPaddingX, int materialPaddingY );
	void rotatePrimitive( 
		int   index, 
		float x, 
		float y, 
		float z );
	void translatePrimitive( 
		int   index, 
		float x, 
		float y, 
		float z );
	void setPrimitiveMaterial( 
		int   index, 
		int   materialId,
		int   materialOffsetX,
		int   materialOffsetY); 
	void getPrimitiveCenter( int index, float& x, float& y, float& z, float& w );
	void setPrimitiveCenter( int index, float  x, float  y, float  z, float  w );

public:

	// ---------- Complex objects ----------
	long addCube( 
		float x, float y, float z, 
		float radius, 
		int   martialId, 
		int   materialPaddingX, int materialPaddingY );

	long CudaKernel::addRectangle( 
		float x, float y, float z, 
		float width, float height,
		float depth,
		int   martialId, 
		int   materialPaddingX, int materialPaddingY );

public:

	// ---------- Lamps ----------
	long addLamp( LampType lampType );
	void setLamp( 
		int index,
		float x, float y, float z, 
      float width, float height,
		float r, float g, float b, float intensity );

public:

	// ---------- Materials ----------
	long addMaterial();
	void setMaterial( 
		int   index,
		float r, float g, float b, 
		float reflection, 
		float refraction,
		int   textured,
		float transparency,
		int   textureId,
		float specValue, float specPower, float specCoef,
		float innerIllumination );

public:

	// ---------- Camera ----------
	void setCamera( 
		float4 eye, float4 dir, float4 angles );

public:

	// ---------- Textures ----------
	void setTexture(
		int   index,
		char* texture );

	long addTexture( 
		const std::string& filename );

public:

   int getImageSize() { return m_imageWidth*m_imageHeight*4; } // Image depth: 32 bits
   int getImageWidth() { return m_imageWidth; }
   int getImageHeight() { return m_imageHeight; }

#ifdef USE_KINECT
public:
	// ---------- Kinect ----------

	long updateSkeletons( 
		double center_x, double  center_y, double  center_z, 
		double size,
		double radius,       int materialId,
		double head_radius,  int head_materialId,
		double hands_radius, int hands_materialId,
		double feet_radius,  int feet_materialId);

	bool CudaKernel::getSkeletonPosition( int index, float4& position );
#endif // USE_KINECT

public:

	int getNbActivePrimitives() { return m_nbActivePrimitives; };
	int getNbActiveLamps()      { return m_nbActiveLamps; };
	int getNbActiveMaterials()  { return m_nbActiveMaterials; };

private:

	float4 normalVector( float4 v1, float4 v2 );
	void   normalizeVector( float4 &v );
	float  vectorLength( const float4 v );
	float  dotProduct( float4 v1, float4 v2 );
	char*  loadFromFile( const std::string& filename, size_t& length );

private:

	// Host
	Primitive* m_hPrimitives;
	Lamp*		  m_hLamps;
	Material*  m_hMaterials;
	char*      m_hTextures;
	float4*    m_hDepthOfField;
	int*		  m_hLevels;
	float*	  m_hRandoms;

private:

   int         m_imageWidth;
   int         m_imageHeight;
	int			m_nbActivePrimitives;
	int			m_nbActiveLamps;
	int			m_nbActiveMaterials;
	int			m_nbActiveTextures;
	float4		m_viewPos;
	float4		m_viewDir;
	float4		m_angles;
	int			m_levelsSize;

private:

	int			m_initialDraft;
	int			m_draft;
	bool		   m_texturedTransfered;

private:

   // Runtime kernel execution parameters
   dim3 m_blockSize;

#ifdef USE_KINECT
	// Kinect declarations
private:

	char*              m_hVideo;
	char*              m_hDepth;
	HANDLE             m_skeletons;
	HANDLE             m_hNextDepthFrameEvent; 
	HANDLE             m_hNextVideoFrameEvent;
	HANDLE             m_hNextSkeletonEvent;
	HANDLE             m_pVideoStreamHandle;
	HANDLE             m_pDepthStreamHandle;
	NUI_SKELETON_FRAME m_skeletonFrame;

	long               m_skeletonIndex;
	long               m_skeletonsBody;
	long               m_skeletonsLamp;
#endif // USE_KINECT

};
