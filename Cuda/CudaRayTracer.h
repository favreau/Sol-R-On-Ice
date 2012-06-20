/* 
* OpenCL Raytracer
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

#include "CudaDataTypes.h"

extern "C" void initialize_scene( 
	int width, int height, int nbPrimitives, int nbLamps, int nbMaterials, int nbTextures, int nbLevels );

extern "C" void finalize_scene();

extern "C" void h2d_scene( 
	Primitive*  primitives, int nbPrimitives,
	Lamp*       lamps,      int nbLamps );

extern "C" void h2d_materials( 
	Material*  materials, int nbActiveMaterials,
	char*      textures,  int nbActiveTextures,
   float*     randoms,   int nbRandoms,
   int*       levels,    int levelSize );

extern "C" void d2h_bitmap( unsigned char* bitmap, int size );

extern "C" void cudaRender(
   dim3 blockSize,
	int nbPrimitives, int nbLamps,
	float4 origin,
	float4 target,
	float4 angles,
	int    width, 
	int    height, 
	float  pointOfFocus,
	int    draft,
	float  transparentColor,
	float  timer);

#ifdef USE_KINECT
extern "C" void h2d_kinect( 
   char* video, int videoSize,
   char* depth, int depthSize );
#endif // USE_KINECT
