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

#include "TextureMapping.cuh"
#include "Colors.cuh"

/*
Lamp Intersection
Lamp         : Object on which we want to find the intersection
origin       : Origin of the ray
orientation  : Orientation of the ray
intersection : Resulting intersection
invert       : true if we want to check the front of the sphere, false for the 
back
returns true if there is an intersection, false otherwise
*/
__device__ bool lampIntersection( 
	Lamp    lamp, 
	float4  origin, 
	float4  ray, 
	float4  O_C,
	float4& intersection)
{
	float si_A = 2.f*(ray.x*ray.x + ray.y*ray.y + ray.z*ray.z);
	if ( si_A == 0.f ) return false;

	bool  si_b1 = false; 
	float si_B = 2.f*(O_C.x*ray.x + O_C.y*ray.y + O_C.z*ray.z);
	float si_C = O_C.x*O_C.x+O_C.y*O_C.y+O_C.z*O_C.z-lamp.center.w*lamp.center.w;
	float si_radius = si_B*si_B-2.f*si_A*si_C;
	float si_t1 = (-si_B-sqrt(si_radius))/si_A;

	if( si_t1>0.f ) 
	{
		intersection = origin+si_t1*ray;
		si_b1 = true;
	}
	return si_b1;
}

/**
Sphere Intersection
primitive    : Object on which we want to find the intersection
origin       : Origin of the ray
orientation  : Orientation of the ray
intersection : Resulting intersection
invert       : true if we want to check the front of the sphere, false for the 
back
returns true if there is an intersection, false otherwise
*/
__device__ bool sphereIntersection(
   Primitive sphere, 
   Material* materials, 
   char*     textures, 
   float4    origin, 
   float4    ray, 
   float     timer,
   float4&   intersection,
   float4&   normal,
   float&    shadowIntensity,
   float     transparentColor
   ) 
{
   // solve the equation sphere-ray to find the intersections
   bool result = false;

   float4 O_C = origin - sphere.p0;
   normalizeVector(ray);
   if(( dotProduct( O_C, ray ) > 0.f ) && (vectorLength(O_C) > sphere.size.x)) return false;

   float a = 2.f*dotProduct(ray,ray);
   float b = 2.f*dotProduct(O_C,ray);
   float c = dotProduct(O_C,O_C) - (sphere.size.x*sphere.size.x);
   float d = b*b-2.f*a*c;
   if( d>0.f && a != 0.f) 
   {
      float r = sqrt(d);
      float t1 = (-b-r)/a;
      float t2 = (-b+r)/a;
      float ta = (t1<t2) ? t1 : t2;

      if( ta > 0.1f ) 
      {
         // First intersection
         intersection = origin+ta*ray;
         result = true ;
      }

      if( !result )
      {
         float tb = (t2<t1) ? t1 : t2;
         if( tb > 0.1f ) 
         {
            // Second intersection
            intersection = origin+tb*ray;
            result = true ;
         }
      }

      if( result ) 
      { 
         // Transparency
         if (materials[sphere.materialId].texture.y != NO_TEXTURE ) 
         {
            float4 color = sphereMapping(sphere, materials, sphere.materialId, textures, intersection );
            result = ((color.x+color.y+color.z) >= transparentColor ); 
         }

         // Compute normal vector
         normal = intersection-sphere.p0;
         normal.w = 0.f;
         shadowIntensity = 1.f-materials[sphere.materialId].transparency;
   
         if( materials[sphere.materialId].texture.x != 0 ) 
         {
            // Procedural texture
            float4 newCenter;
            newCenter.x = sphere.p0.x + 5.f*cos(timer*0.058f+intersection.x);
            newCenter.y = sphere.p0.y + 5.f*sin(timer*0.085f+intersection.y);
            newCenter.z = sphere.p0.z + 5.f*sin(cos(timer*0.124f+intersection.z));
            normal  = intersection - newCenter;
         }

         //float4 loi = intersection-origin;
         //if( dotProduct(loi,normal) <= 0.f ) normal = -normal;
         normalizeVector(normal);
      }
   }

#if 0
   // Soft Shadows
   if( result && computingShadows ) 
   {
      float4 O_R;
      O_R.x = ray.x-origin.x;
      O_R.y = ray.y-origin.y;
      O_R.z = ray.z-origin.z;

      normalizeVector(O_R);
      shadowIntensity = dotProduct(O_R, (normal));
      shadowIntensity = (shadowIntensity>1.f) ? 1.f : shadowIntensity;
      shadowIntensity = (shadowIntensity<0.f) ? 0.f : shadowIntensity;
   } 
#endif // 0
   return result;
}

/*
Cylinder Intersection
primitive    : Object on which we want to find the intersection
origin       : Origin of the ray
orientation  : Orientation of the ray
intersection : Resulting intersection
invert       : true if we want to check the front of the sphere, false for the 
back
returns true if there is an intersection, false otherwise
*/
__device__ bool cylinderIntersection( 
	Primitive cylinder, 
	Material* materials, 
	char*     textures,
   int*      levels,
	float4    origin, 
	float4    ray, 
	float     timer,
	float4&   intersection,
	float4&   normal,
	float&    shadowIntensity,
	float     transparentColor ) 
{
	// solve the equation sphere-ray to find the intersections
	bool result = false;

#if 0
	// Top
	if(!result && ray.y<0.f && origin.y>(cylinder.p0.y+cylinder.size.y)) 
	{
		intersection.y = cylinder.p0.y+cylinder.size.y;
		float y = origin.y-cylinder.p0.y-cylinder.size.y;
		intersection.x = origin.x+y*ray.x/-ray.y;
		intersection.z = origin.z+y*ray.z/-ray.y;
		intersection.w = 1.f; // 1 for top, -1 for bottom

		float4 v=intersection-cylinder.p0;
		v.y = 0.f;
		result = (vectorLength(v)<cylinder.size.x);

		(normal).x =  0.f;
		(normal).y =  1.f;
		(normal).z =  0.f;
	}

	// Bottom
	if( !result && ray.y>0.f && origin.y<(cylinder.p0.y - cylinder.size.y) ) 
	{
		intersection.y = cylinder.p0.y - cylinder.size.y;
		float y = origin.y - cylinder.p0.y + cylinder.size.y;
		intersection.x = origin.x+y*ray.x/-ray.y;
		intersection.z = origin.z+y*ray.z/-ray.y;
		intersection.w = -1.f; // 1 for top, -1 for bottom

		float4 v=intersection-cylinder.p0;
		v.y = 0.f;
		result = (vectorLength(v)<cylinder.size.x);

		(normal).x =  0.f;
		(normal).y = -1.f;
		(normal).z =  0.f;
	}
#endif // 0

	if( !result ) 
	{
		float4 O_C = origin - cylinder.p0;
		O_C.x = 0.f;
		if(( dotProduct( O_C, ray ) > 0.f ) && (vectorLength(O_C) > cylinder.p0.w)) return false;

		float a = 2.f * ( ray.y*ray.y + ray.z*ray.z );
		float b = 2.f*((origin.y-cylinder.p0.y)*ray.y + (origin.z-cylinder.p0.z)*ray.z);
		float c = O_C.y*O_C.y + O_C.z*O_C.z - cylinder.size.x*cylinder.size.x;

      float d = b*b-2.f*a*c;

		// Cylinder
		if ( d >= 0.f && a != 0.f) 
      {
		   float r = sqrt(d);
		   float t1 = (-b-r)/a;
		   float t2 = (-b+r)/a;
		   float ta = (t1<t2) ? t1 : t2;
		   float tb = (t2<t1) ? t1 : t2;

		   if( ta > 0.001f )
		   {
			   intersection = origin+ta*ray;
			   intersection.w = 0.f;

			   result = ( fabs(intersection.x - cylinder.p0.x) <= cylinder.size.x );
		   }

		   if( !result && tb > 0.001f ) 
		   {
			   intersection = origin+tb*ray;
			   intersection.w = 0.f;

			   result = ( fabs(intersection.x - cylinder.p0.x) <= cylinder.size.x );
		   }
      }
	}

	// Normal to surface
	if( result ) 
	{
		normal   = intersection-cylinder.p0;
		normal.x = 0.f;
		normal.w = 0.f;

      //float4 loi = intersection-origin;
      //if( dotProduct(loi,normal) < 0.f ) normal = -normal;
		shadowIntensity = 1.f-materials[cylinder.materialId].transparency;
		if( materials[cylinder.materialId].texture.x ) 
		{
			float4 newCenter;
			newCenter.x = cylinder.p0.x + 5.f*cos(timer*0.58f+intersection.x);
			newCenter.y = cylinder.p0.y + 5.f*sin(timer*0.85f+intersection.y) + intersection.y;
			newCenter.z = cylinder.p0.z + 5.f*sin(cos(timer*1.24f+intersection.z));
			normal = intersection-newCenter;
		}
		normalizeVector( normal );
      result = true;
	}

#if 0
	// Soft Shadows
	if( result && computingShadows ) 
	{
		float4 normal = normalToSurface( cylinder, intersection, depth, materials, timer ); // Normal is computed twice!!!
		normalizeVector(ray );
		normalizeVector(normal);
		shadowIntensity = 5.f*fabs(dotProduct(-ray ,normal));
		shadowIntensity = (shadowIntensity>1.f) ? 1.f : shadowIntensity;
	} 
#endif // 0
	return result;
}

/*
Checkboard Intersection
primitive    : Object on which we want to find the intersection
origin       : Origin of the ray
orientation  : Orientation of the ray
intersection : Resulting intersection
invert       : true if we want to check the front of the sphere, false for the 
back
returns true if there is an intersection, false otherwise
*/
__device__ bool planeIntersection( 
	Primitive primitive,
	Material* materials,
	char*     textures,
   int*      levels,
	float4    origin, 
	float4    ray, 
	bool      reverse,
	float&    shadowIntensity,
	float4&   intersection,
	float4&   normal,
	float     transparentColor,
	float     timer)
{ 
	bool collision = false;

	float reverted = reverse ? -1.f : 1.f;
	switch( primitive.type ) 
	{
   case ptMagicCarpet:
	case ptCheckboard:
		{
			intersection.y = primitive.p0.y;
			float y = origin.y-primitive.p0.y;
			if( reverted*ray.y<0.f && reverted*origin.y>reverted*primitive.p0.y) 
			{
				intersection.x = origin.x+y*ray.x/-ray.y;
				intersection.z = origin.z+y*ray.z/-ray.y;
				collision = 
					fabs(intersection.x - primitive.p0.x) < primitive.size.x &&
					fabs(intersection.z - primitive.p0.z) < primitive.size.y;
				(normal).x =  0.f;
				(normal).y =  1.f;
				(normal).z =  0.f;
			}
			break;
		}
	case ptXZPlane:
		{
			float y = origin.y-primitive.p0.y;
			if( reverted*ray.y<0.f && reverted*origin.y>reverted*primitive.p0.y) 
			{
				(normal).x =  0.f;
				(normal).y =  1.f;
				(normal).z =  0.f;
				intersection.x = origin.x+y*ray.x/-ray.y;
				intersection.y = primitive.p0.y;
				intersection.z = origin.z+y*ray.z/-ray.y;
				collision = 
					fabs(intersection.x - primitive.p0.x) < primitive.size.x &&
					fabs(intersection.z - primitive.p0.z) < primitive.size.y;
			}
			if( !collision && reverted*ray.y>0.f && reverted*origin.y<reverted*primitive.p0.y) 
			{
				(normal).x =  0.f;
				(normal).y =  -1.f;
				(normal).z =  0.f;
				intersection.x = origin.x+y*ray.x/-ray.y;
				intersection.y = primitive.p0.y;
				intersection.z = origin.z+y*ray.z/-ray.y;
				collision = 
					fabs(intersection.x - primitive.p0.x) < primitive.size.x &&
					fabs(intersection.z - primitive.p0.z) < primitive.size.y;
			}
			break;
		}
	case ptYZPlane:
		{
			float x = origin.x-primitive.p0.x;
			if( reverted*ray.x<0.f && reverted*origin.x>reverted*primitive.p0.x ) 
			{
				intersection.x = primitive.p0.x;
				intersection.y = origin.y+x*ray.y/-ray.x;
				intersection.z = origin.z+x*ray.z/-ray.x;
				collision = 
					fabs(intersection.y - primitive.p0.y) < primitive.size.y &&
					fabs(intersection.z - primitive.p0.z) < primitive.size.x;
				(normal).x =  1.f;
				(normal).y =  0.f;
				(normal).z =  0.f;
			}
			if( !collision && reverted*ray.x>0.f && reverted*origin.x<reverted*primitive.p0.x ) 
			{
				intersection.x = primitive.p0.x;
				intersection.y = origin.y+x*ray.y/-ray.x;
				intersection.z = origin.z+x*ray.z/-ray.x;
				collision = 
					fabs(intersection.y - primitive.p0.y) < primitive.size.y &&
					fabs(intersection.z - primitive.p0.z) < primitive.size.x;
				(normal).x = -1.f;
				(normal).y =  0.f;
				(normal).z =  0.f;
			}
			break;
		}
	case ptXYPlane:
		{
			float z = origin.z-primitive.p0.z;
			if( reverted*ray.z<0.f && reverted*origin.z>reverted*primitive.p0.z) 
			{
				intersection.z = primitive.p0.z;
				intersection.x = origin.x+z*ray.x/-ray.z;
				intersection.y = origin.y+z*ray.y/-ray.z;
				collision = 
					fabs(intersection.x - primitive.p0.x) < primitive.size.x &&
					fabs(intersection.y - primitive.p0.y) < primitive.size.y;
				(normal).x =  0.f;
				(normal).y =  0.f;
				(normal).z =  1.f;
			}
			if( !collision && reverted*ray.z>0.f && reverted*origin.z<reverted*primitive.p0.z )
			{
				intersection.z = primitive.p0.z;
				intersection.x = origin.x+z*ray.x/-ray.z;
				intersection.y = origin.y+z*ray.y/-ray.z;
				collision = 
					fabs(intersection.x - primitive.p0.x) < primitive.size.x &&
					fabs(intersection.y - primitive.p0.y) < primitive.size.y;
				(normal).x =  0.f;
				(normal).y =  0.f;
				(normal).z = -1.f;
			}
			break;
		}
	case ptCamera:
		{
			if( reverted*ray.z>0.f && reverted*origin.z<reverted*primitive.p0.z )
			{
				intersection.z = primitive.p0.z;
				float z = origin.z-primitive.p0.z;
				intersection.x = origin.x+z*ray.x/-ray.z;
				intersection.y = origin.y+z*ray.y/-ray.z;
				collision =
					fabs(intersection.x - primitive.p0.x) < primitive.size.x &&
					fabs(intersection.y - primitive.p0.y) < primitive.size.y;
				(normal).x =  0.f;
				(normal).y =  0.f;
				(normal).z = -1.f;
			}
			break;
		}
	}

	if( collision ) 
	{
		shadowIntensity = 1.f;
		float4 color;
      color = materials[primitive.materialId].color;
      if(materials[primitive.materialId].texture.y != NO_TEXTURE)
		{
			color = cubeMapping(primitive, materials, textures, intersection );
		}

		if( ((color.x+color.y+color.z)/3.f) <= transparentColor) 
		{
			collision = false;
		}
		else 
		{
			shadowIntensity = ((color.x+color.y+color.z)/3.f*(1.f-materials[primitive.materialId].transparency));
		}
	}
	return collision;
}

__device__ bool triangleIntersection( 
	Primitive          triangle, 
	float4             origin, 
	float4             ray,
	float              timer,
	float4&            intersection,
	float4&            normal,
	bool               computingShadows,
	float&             shadowIntensity,
	float              transparentColor
	) 
{
	bool result = false;

	float lD = -triangle.p0.x*(triangle.p1.y*triangle.p2.z - triangle.p2.y*triangle.p1.z)
		-triangle.p1.x*(triangle.p2.y*triangle.p0.z - triangle.p0.y*triangle.p2.z)
		-triangle.p2.x*(triangle.p0.y*triangle.p1.z - triangle.p1.y*triangle.p0.z);

	float d = triangle.normal.x*ray.x + triangle.normal.y*ray.y + triangle.normal.z*ray.z;

	d += (d==0.f) ? 0.01f : 0.f;

	float t = -(triangle.normal.x*origin.x + triangle.normal.y*origin.y + triangle.normal.z*origin.z + lD) / d;

	if(t > 0.f)// Triangle in front of the ray
	{
		float4 i = origin+t*ray;

		// 1st side
		float4 I = i - triangle.p0;
		if (dotProduct(triangle.v0,I) <= 0.f)
		{
			// 1st side OK
			I = i - triangle.p1;
			if (dotProduct(triangle.v1,I) <= 0.f)
			{
				// 2nd side OK
				I = i - triangle.p2;
				if (dotProduct(triangle.v2,I) <= 0.f)
				{
					// 1st side OK
					intersection = i;
					(normal) = triangle.normal;
					result = true;
				}
			}
		}
	}
	return result;
}
