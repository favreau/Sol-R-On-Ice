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
__device__ void makeOpenGLColor( 
   float4 color,
   char*  bitmap,
   int    index)
{
   int mdc_index = index*gColorDepth; 
   bitmap[mdc_index  ] = (char)(color.x*255.f); // Red
   bitmap[mdc_index+1] = (char)(color.y*255.f); // Green
   bitmap[mdc_index+2] = (char)(color.z*255.f); // Blue
   bitmap[mdc_index+3] = 0.f; //(char)(color.w*255.f); // Alpha
}

/*
* Colors
*/
__device__ float4 objectColorAtIntersection( 
	Primitive primitive, 
	Material* materials,
	char*     textures,
   char*     kinectVideo,
   int*      levels,
	float4    intersection,
	float     timer, 
	bool      back )
{
	float4 colorAtIntersection = materials[primitive.materialId].color;
	switch( primitive.type ) 
	{
	case ptCylinder:
		{
         if(materials[primitive.materialId].texture.y != NO_TEXTURE)
         {
				colorAtIntersection = magicCylinderMapping(primitive, materials, textures, intersection, levels, timer);
         }
			break;
		}
	case ptSphere:
		{
         if(materials[primitive.materialId].texture.y != NO_TEXTURE)
         {
				colorAtIntersection = sphereMapping(primitive, materials, primitive.materialId, textures, intersection );
         }
			break;
		}
	case ptTriangle:
		break;
   case ptMagicCarpet:
      {
			if( materials[primitive.materialId].texture.y != NO_TEXTURE ) 
			{
				colorAtIntersection = magicCarpetMapping( primitive, materials, textures, intersection, levels, timer );
			}
         break;
      }
	case ptCheckboard :
		{
			if( materials[primitive.materialId].texture.y != NO_TEXTURE ) 
			{
				colorAtIntersection = cubeMapping( primitive, materials, textures, intersection );
			}
			else 
			{
				int x = gMaxViewDistance + ((intersection.x - primitive.p0.x)/primitive.p0.w*primitive.materialInfo.x);
				int z = gMaxViewDistance + ((intersection.z - primitive.p0.z)/primitive.p0.w*primitive.materialInfo.y);
				if(x%2==0) 
				{
					if (z%2==0) 
					{
						colorAtIntersection.x = 1.f-colorAtIntersection.x;
						colorAtIntersection.y = 1.f-colorAtIntersection.y;
						colorAtIntersection.z = 1.f-colorAtIntersection.z;
					}
				}
				else 
				{
					if (z%2!=0) 
					{
						colorAtIntersection.x = 1.f-colorAtIntersection.x;
						colorAtIntersection.y = 1.f-colorAtIntersection.y;
						colorAtIntersection.z = 1.f-colorAtIntersection.z;
					}
				}
			}
			break;
		}
	case ptXYPlane:
	case ptYZPlane:
	case ptXZPlane:
		{
			if( materials[primitive.materialId].texture.y != NO_TEXTURE ) 
			{
				colorAtIntersection = cubeMapping( primitive, materials, textures, intersection );
			}
			break;
		}
	case ptCamera:
		{
			colorAtIntersection = materials[primitive.materialId].color;
			int x = (intersection.x-primitive.p0.x)*primitive.materialInfo.x+gKinectVideoWidth/2;
			int y = gKinectVideoHeight/2 - (intersection.y - primitive.p0.y)*primitive.materialInfo.y;

			x = (x+gKinectVideoWidth)%gKinectVideoWidth;
			y = (y+gKinectVideoHeight)%gKinectVideoHeight;

			if( x>=0 && x<gKinectVideoWidth && y>=0 && y<gKinectVideoHeight ) 
			{
				int index = (y*gKinectVideoWidth+x)*gKinectVideo;
				unsigned char r = kinectVideo[index+2];
				unsigned char g = kinectVideo[index+1];
				unsigned char b = kinectVideo[index+0];
				colorAtIntersection.x = r/256.f;
				colorAtIntersection.y = g/256.f;
				colorAtIntersection.z = b/256.f;
			}
			break;
		}
	}
	return colorAtIntersection;
}
