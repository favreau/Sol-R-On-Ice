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
#include "Consts.h"

// Scene information
struct SceneInfo
{
   int    width;
   int    height;
   float  draft;
   float  transparentColor;
   bool   shadowsEnabled;
   float  viewDistance;
   float  shadowIntensity;
   int    nbRayIterations;
   float4 backgroundColor;
   bool   supportFor3DVision;
   float  width3DVision;
   bool   renderBoxes;
};

struct Ray 
{
   float4 origin;
   float4 direction;
   float4 inv_direction;
   int sign[3];
};

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
   ptMagicCarpet = 8,
   ptEnvironment = 9
};

enum LampType
{
   ltSphere  = 0,
   ltXYPlane = 1,
   ltXZPlane = 2,
   ltSpot    = 3
};

struct Material
{
	float4 color;
   float4 specular;    // x: value, y: power, w: coef, z: inner illumination
   float  reflection;
	float  refraction;
   float  transparency;
	bool   textured;
   int    textureId;
};

struct BoundingBox
{
   float4 parameters[2];
   int    nbPrimitives;
   int    primitiveIndex[NB_MAX_PRIMITIVES_PER_BOX];
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
   float4   direction;
   LampType lampType;
   float    intensity;
};

// Post processing effect
struct DepthOfFieldInfo
{
   bool   enabled;
   float  pointOfFocus;
   float  strength;
   int    iterations;
};

