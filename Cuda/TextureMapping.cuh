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

#include "Scene.cuh"

// ________________________________________________________________________________
__device__ float4 sphereMapping( 
   Primitive primitive,
   Material* materials,
   int       materialId,
   char*     textures,
   float4    intersection)
{
   float4 result = materials[primitive.materialId].color;
   int x = gTextureOffset+(intersection.x-primitive.p0.x+primitive.size.x)*primitive.materialInfo.x;
   int y = gTextureOffset+(intersection.y-primitive.p0.y+primitive.size.y)*primitive.materialInfo.y;

   x = x%gTextureWidth;
   y = y%gTextureHeight;

   if( x>=0 && x<gTextureWidth && y>=0 && y<gTextureHeight )
   {
      int index = (materials[materialId].texture.y*gTextureWidth*gTextureHeight + y*gTextureWidth+x)*gTextureDepth;
      unsigned char r = textures[index  ];
      unsigned char g = textures[index+1];
      unsigned char b = textures[index+2];
      result.x = r/256.f;
      result.y = g/256.f;
      result.z = b/256.f;
   }
   return result;
}

// ________________________________________________________________________________
__device__ float4 cubeMapping( 
   Primitive primitive, 
   Material* materials,
   char*     textures,
   float4    intersection)
{
   float4 result = materials[primitive.materialId].color;
   int x = ((primitive.type == ptCheckboard) || (primitive.type == ptXZPlane) || (primitive.type == ptXYPlane))  ? 
      gTextureOffset+(intersection.x-primitive.p0.x+primitive.size.x)*primitive.materialInfo.x :
      gTextureOffset+(intersection.z-primitive.p0.z+primitive.size.x)*primitive.materialInfo.x;

   int y = ((primitive.type == ptCheckboard)  || (primitive.type == ptXZPlane)) ? 
      gTextureOffset+(intersection.z+primitive.p0.z+primitive.size.y)*primitive.materialInfo.y :
      gTextureOffset+(intersection.y-primitive.p0.y+primitive.size.y)*primitive.materialInfo.y;

   x = x%gTextureWidth;
   y = y%gTextureHeight;

   if( x>=0 && x<gTextureWidth && y>=0 && y<gTextureHeight )
   {
      int index = (materials[primitive.materialId].texture.y*gTextureWidth*gTextureHeight + y*gTextureWidth+x)*gTextureDepth;
      unsigned char r = textures[index];
      unsigned char g = textures[index+1];
      unsigned char b = textures[index+2];
      result.x = r/256.f;
      result.y = g/256.f;
      result.z = b/256.f;
   }
   return result;
}

// ________________________________________________________________________________
__device__ float4 magicCarpetMapping( 
   Primitive primitive, 
   Material* materials,
   char*     textures,
   float4    intersection,
   int*      levels,
   float     timer)
{
   float4 result = materials[primitive.materialId].color;
	int x = gTextureOffset+(intersection.x-primitive.p0.x+primitive.size.x)*primitive.materialInfo.x*5.f;
	int y = gTextureOffset+(intersection.z+timer-primitive.p0.z+primitive.size.y)*primitive.materialInfo.y*50.f;

   x = x%gTextureWidth;
   y = y%gTextureHeight;

   if( x>=0 && x<gTextureWidth && y>=0 && y<gTextureHeight )
   {
      // Level management
		int tid_x = (intersection.x-primitive.p0.x+primitive.size.x      )/(primitive.size.x/2.5f);
		int tid_y = (intersection.z-primitive.p0.z+primitive.size.y+timer)/(primitive.size.y/25.f);
		int tid = tid_x+tid_y*5;
		tid = tid%5000;
      int index = (levels[tid]*gTextureWidth*gTextureHeight + y*gTextureWidth+x)*gTextureDepth;
      unsigned char r = textures[index];
      unsigned char g = textures[index+1];
      unsigned char b = textures[index+2];
      result.x = r/256.f;
      result.y = g/256.f;
      result.z = b/256.f;
   }
   return result;
}

// ________________________________________________________________________________
__device__ float4 magicCylinderMapping( 
   Primitive primitive, 
   Material* materials,
   char*     textures,
   float4    intersection,
   int*      levels,
   float     timer)
{
   float4 result = materials[primitive.materialId].color;
   
	int x = gTextureOffset+(intersection.x-      primitive.p0.x+primitive.size.x)*primitive.materialInfo.x*5.f;
	int y = gTextureOffset+(intersection.z+timer-primitive.p0.z+primitive.size.y)*primitive.materialInfo.y*50.f;

   x = x%gTextureWidth;
   y = y%gTextureHeight;

   if( x>=0 && x<gTextureWidth && y>=0 && y<gTextureHeight )
   {
		int tid_x = (intersection.x-primitive.p0.x+primitive.size.x      )/(primitive.size.x/2.5f);
		int tid_y = (intersection.z-primitive.p0.z+primitive.size.y+timer)/(primitive.size.y/25.f);
		int tid = tid_x+tid_y*5;
		tid = tid%5000;
      int index = (levels[tid]*gTextureWidth*gTextureHeight + y*gTextureWidth+x)*gTextureDepth;
      unsigned char r = textures[index  ];
      unsigned char g = textures[index+1];
      unsigned char b = textures[index+2];
      result.x = r/256.f;
      result.y = g/256.f;
      result.z = b/256.f;
   }
   return result;
}

