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

#include <vector_types.h>

// Enums
enum PrimitiveType 
{
	ptSphere      = 0,
	ptTriangle    = 1,
	ptCheckboard  = 2,
	ptCamera      = 3,
	ptXYPlane     = 4,
	ptYZPlane     = 5,
	ptXZPlane     = 6,
	ptCylinder    = 7,
   ptMagicCarpet = 8
};

enum LampType
{
   ltSphere  = 0,
   ltXYPlane = 1,
   ltXZPlane = 2
};

struct Material
{
	float4 color;
   float4 specular;    // x: value, y: power, w: coef, z: inner illumination
   float  reflection;
	float  refraction;
   float  transparency;
	int2   texture;
   float  bonus;
};

struct Primitive
{
	float4 p0;
	float4 p1;
	float4 p2;
	float4 v0;
	float4 v1;
	float4 v2;
	float4 normal;
	float4 rotation;
	float4 size;
	PrimitiveType type;
	int    materialId;
	float2 materialInfo;
};

struct Lamp
{
   float4   center;
   float4   color;
   LampType lampType;
   float    intensity;
};

// Constants
#define NO_MATERIAL -1
#define NO_TEXTURE  -1
#define gColorDepth    4
#define gTextureOffset 0.f
#define gTextureWidth  256
#define gTextureHeight 256
#define gTextureDepth  3

#define gKinectVideoWidth  640
#define gKinectVideoHeight 480
#define gKinectVideo       4

#define gKinectDepthWidth  320
#define gKinectDepthHeight 240
#define gKinectDepth       2
