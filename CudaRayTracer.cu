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

// System
#include <iostream>

// Cuda
#include <cuda_runtime_api.h>
#if CUDART_VERSION>=5000
   #include <helper_cuda.h>
   #include <helper_math.h>
#else
   #include <cutil_inline.h>
   #include <cutil_math.h>
#endif

// Project
#include "CudaDataTypes.h"

// Cuda Runtime paramters
#define BLOCK_SIZE 16

// Globals
#define gNbIterations 20
#define gNbMaxShadowCollisions 10
#define gSoftShadowPrecision   1.f

// Device arrays
Primitive*   d_primitives; 
BoundingBox* d_boundingBoxes; 
Lamp*        d_lamps;
Material*    d_materials;
char*        d_textures;
float*       d_randoms;
int*         d_levels;
float4*      d_postProcessingBuffer;
char*        d_bitmap; 

#ifdef USE_KINECT
char*        d_kinectVideo;
char*        d_kinectDepth;
#endif // USE_KINECT

// ________________________________________________________________________________
__device__ inline float vectorLength( float4 vector )
{
   return sqrt( vector.x*vector.x + vector.y*vector.y + vector.z*vector.z );
}

// ________________________________________________________________________________
__device__ inline void normalizeVector( float4& v )
{
   v /= vectorLength( v );
}

// ________________________________________________________________________________
__device__ inline void saturateVector( float4& v )
{
   v.x = (v.x>1.f) ? 1.f : v.x;
   v.y = (v.y>1.f) ? 1.f : v.y; 
   v.z = (v.z>1.f) ? 1.f : v.z;
   v.w = (v.w>1.f) ? 1.f : v.w;
}

// ________________________________________________________________________________
__device__ inline float dotProduct( float4 &v1, float4& v2 )
{
   return ( v1.x*v2.x + v1.y*v2.y + v1.z*v2.z);
}

/*
________________________________________________________________________________
incident  : le vecteur normal inverse a la direction d'incidence de la source 
lumineuse
normal    : la normale a l'interface orientee dans le materiau ou se propage le 
rayon incident
reflected : le vecteur normal reflechi
________________________________________________________________________________
*/
__device__ inline void vectorReflection( float4& r, float4& i, float4& n )
{
   r = i-2.f*dotProduct(i,n)*n;
}

/*
________________________________________________________________________________
incident: le vecteur norm? inverse ? la direction d?incidence de la source 
lumineuse
n1      : index of refraction of original medium
n2      : index of refraction of new medium
________________________________________________________________________________
*/
__device__ inline void vectorRefraction( 
   float4& refracted, 
   float4 incident, 
   float n1, 
   float4 normal, 
   float n2 )
{
   if( n1 != n2 ) 
   {
      float r = n1/n2;
      float cosI = dotProduct( incident, normal );
      float cosT2 = 1.0f - r*r*(1.0f - cosI*cosI);
      if( cosT2 > 0.01f ) 
      {
         refracted = r*incident + (r*cosI-sqrt( fabs(cosT2) ))*normal;
      }
   }
   else 
   {
      refracted = incident;
   }
}

/*
________________________________________________________________________________
__v : Vector to rotate
__c : Center of rotations
__a : Angles
________________________________________________________________________________
*/
__device__ inline void vectorRotation( float4& vector, float4 center, float4 angles )
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

Compute ray attributes
________________________________________________________________________________
*/
__device__ inline void computeRayAttributes(Ray& ray)
{
   ray.inv_direction.x = 1.f/ray.direction.x;
   ray.inv_direction.y = 1.f/ray.direction.y;
   ray.inv_direction.z = 1.f/ray.direction.z;
   ray.sign[0] = (ray.inv_direction.x < 0);
   ray.sign[1] = (ray.inv_direction.y < 0);
   ray.sign[2] = (ray.inv_direction.z < 0);
}

/*
________________________________________________________________________________

Convert float4 into OpenGL RGB color
________________________________________________________________________________
*/
__device__ void makeOpenGLColor( 
   float4 color,
   char*  bitmap,
   int    index)
{
   int mdc_index = index*gColorDepth; 
   bitmap[mdc_index  ] = (char)(color.x*255.f); // Red
   bitmap[mdc_index+1] = (char)(color.y*255.f); // Green
   bitmap[mdc_index+2] = (char)(color.z*255.f); // Blue
   bitmap[mdc_index+3] = (char)(color.w*255.f); // Alpha
}

/*
________________________________________________________________________________

Sphere texture Mapping
________________________________________________________________________________
*/
__device__ float4 sphereMapping( 
   Primitive& primitive,
   Material*  materials,
   char*      textures,
   float4     intersection)
{
   float4 result = materials[primitive.materialId].color;
   int x = gTextureOffset+(intersection.x-primitive.p0.x+primitive.size.y)*primitive.materialInfo.x;
   int y = gTextureOffset+(intersection.y-primitive.p0.y+primitive.size.x)*primitive.materialInfo.y;

   x = x % gTextureWidth;
   y = y % gTextureHeight;

   if( x>=0 && x<gTextureWidth&& y>=0 && y<gTextureHeight )
   {
      int index = (materials[primitive.materialId].textureId*gTextureWidth*gTextureHeight + y*gTextureWidth+x)*gTextureDepth;
      unsigned char r = textures[index  ];
      unsigned char g = textures[index+1];
      unsigned char b = textures[index+2];
      result.x = r/256.f;
      result.y = g/256.f;
      result.z = b/256.f;
   }
   return result; 
}

/*
________________________________________________________________________________

Cube texture mapping
________________________________________________________________________________
*/
__device__ float4 cubeMapping( 
   Primitive& primitive, 
   Material*  materials,
   char*      textures,
   float4     intersection)
{
   float4 result = materials[primitive.materialId].color;
   int x = ((primitive.type == ptCheckboard) || (primitive.type == ptXZPlane) || (primitive.type == ptXYPlane))  ? 
      gTextureOffset+(intersection.x-primitive.p0.x+primitive.size.x)*primitive.materialInfo.x :
   gTextureOffset+(intersection.z-primitive.p0.z+primitive.size.x)*primitive.materialInfo.x;

   int y = ((primitive.type == ptCheckboard) || (primitive.type == ptXZPlane)) ? 
      gTextureOffset+(intersection.z+primitive.p0.z+primitive.size.y)*primitive.materialInfo.y :
   gTextureOffset+(intersection.y-primitive.p0.y+primitive.size.y)*primitive.materialInfo.y;

   x = x%gTextureWidth;
   y = y%gTextureHeight;

   if( x>=0 && x<gTextureWidth && y>=0 && y<gTextureHeight )
   {
      int index = (materials[primitive.materialId].textureId*gTextureWidth*gTextureHeight + y*gTextureWidth+x)*gTextureDepth;
      unsigned char r = textures[index];
      unsigned char g = textures[index+1];
      unsigned char b = textures[index+2];
      result.x = r/256.f;
      result.y = g/256.f;
      result.z = b/256.f;
   }
   return result;
}

#if 0
/*
________________________________________________________________________________

Magic Carpet texture mapping
________________________________________________________________________________
*/
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

/*
________________________________________________________________________________

Magic Cylinder texture mapping
________________________________________________________________________________
*/
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
#endif // 0

/*
________________________________________________________________________________

Box intersection
________________________________________________________________________________
*/
__device__ inline bool boxIntersection( 
   BoundingBox& box, 
   Ray          ray,
   float        t0,
   float        t1)
{
   computeRayAttributes( ray );

   float tmin, tmax, tymin, tymax, tzmin, tzmax;

   tmin = (box.parameters[ray.sign[0]].x - ray.origin.x) * ray.inv_direction.x;
   tmax = (box.parameters[1-ray.sign[0]].x - ray.origin.x) * ray.inv_direction.x;
   tymin = (box.parameters[ray.sign[1]].y - ray.origin.y) * ray.inv_direction.y;
   tymax = (box.parameters[1-ray.sign[1]].y - ray.origin.y) * ray.inv_direction.y;

   if ( (tmin > tymax) || (tymin > tmax) ) 
      return false;

   if (tymin > tmin) tmin = tymin;
   if (tymax < tmax) tmax = tymax;
   tzmin = (box.parameters[ray.sign[2]].z - ray.origin.z) * ray.inv_direction.z;
   tzmax = (box.parameters[1-ray.sign[2]].z - ray.origin.z) * ray.inv_direction.z;

   if ( (tmin > tzmax) || (tzmin > tmax) ) 
      return false;

   if (tzmin > tmin) tmin = tzmin;
   if (tzmax < tmax) tmax = tzmax;
   return ( (tmin < t1) && (tmax > t0) );
}

/*
________________________________________________________________________________

Lamp intersection
________________________________________________________________________________
*/
__device__ inline bool lampIntersection( 
   Lamp&   lamp, 
   Ray&    ray, 
   float4  O_C,
   float4& intersection)
{
   float si_A = 2.f*(ray.direction.x*ray.direction.x + ray.direction.y*ray.direction.y + ray.direction.z*ray.direction.z);
   if ( si_A == 0.f ) return false;

   bool  si_b1 = false; 
   float si_B = 2.f*(O_C.x*ray.direction.x + O_C.y*ray.direction.y + O_C.z*ray.direction.z);
   float si_C = O_C.x*O_C.x+O_C.y*O_C.y+O_C.z*O_C.z-lamp.center.w*lamp.center.w;
   float si_radius = si_B*si_B-2.f*si_A*si_C;
   float si_t1 = (-si_B-sqrt(si_radius))/si_A;

   if( si_t1>0.f ) 
   {
      intersection = ray.origin+si_t1*ray.direction;
      si_b1 = true;
   }
   return si_b1;
}

/*
________________________________________________________________________________

Sphere intersection
________________________________________________________________________________
*/
__device__ inline bool sphereIntersection(
   SceneInfo& sceneInfo,
   Primitive& sphere, 
   Material*  materials, 
   char*      textures, 
   Ray&       ray, 
   float      timer,
   float4&    intersection,
   float4&    normal,
   float&     shadowIntensity
   ) 
{
   // solve the equation sphere-ray to find the intersections
   bool result = false;

   float4 O_C = ray.origin - sphere.p0;
   normalizeVector(ray.direction);
   if(( dotProduct( O_C, ray.direction ) > 0.f ) && (vectorLength(O_C) > sphere.size.x)) return false;

   float a = 2.f*dotProduct(ray.direction,ray.direction);
   float b = 2.f*dotProduct(O_C,ray.direction);
   float c = dotProduct(O_C,O_C) - (sphere.size.x*sphere.size.x);
   float d = b*b-2.f*a*c;
   if( d>0.f && a != 0.f) 
   {
      float r = sqrt(d);
      float t1 = (-b-r)/a;
      float t2 = (-b+r)/a;
      float ta = (t1<t2) ? t1 : t2;
      float tb = (t2<t1) ? t1 : t2;
      float4 intersection1;
      float4 intersection2;
      bool i1(false);
      bool i2(false);

      if( ta > 0.1f ) 
      {
         // First intersection
         intersection1 = ray.origin+ta*ray.direction;
         // Transparency
         if (materials[sphere.materialId].textureId != NO_TEXTURE && materials[sphere.materialId].transparency != 0 ) 
         {
            float4 color = sphereMapping(sphere, materials, textures, intersection1 );
            i1 = ((color.x+color.y+color.z) >= sceneInfo.transparentColor ); 
         }
         else
         {
            i1 = true;
         }
      }

      if( tb > 0.1f ) 
      {
         // Second intersection
         intersection2 = ray.origin+tb*ray.direction;
         if (materials[sphere.materialId].textureId != NO_TEXTURE && materials[sphere.materialId].transparency != 0 ) 
         {
            float4 color = sphereMapping(sphere, materials, textures, intersection2 );
            i2 = ((color.x+color.y+color.z) >= sceneInfo.transparentColor ); 
         }
         else
         {
            i2 = true;
         }
      }

      result = i1 || i2;

      if( result ) 
      { 
         if( i1 && i2 )
         {
            float4 O_I1 = intersection1 - ray.origin;
            float4 O_I2 = intersection2 - ray.origin;
            intersection = ( vectorLength(O_I1)<vectorLength(O_I2)) ? intersection1 : intersection2;
         }
         else 
         {
            intersection = i1 ? intersection1 : intersection2;
         }

         // Compute normal vector
         normal = intersection-sphere.p0;
         normal.w = 0.f;
         shadowIntensity = sceneInfo.shadowIntensity*1.f-materials[sphere.materialId].transparency;

         if( materials[sphere.materialId].textured ) 
         {
            // Procedural texture
            float4 newCenter;
            newCenter.x = sphere.p0.x + 5.f*cos(intersection.x);
            newCenter.y = sphere.p0.y + 5.f*sin(intersection.y);
            newCenter.z = sphere.p0.z + 5.f*sin(cos(intersection.z));
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
      shadowIntensity = dotProduct(O_R, normal);
      shadowIntensity = (shadowIntensity>1.f) ? 1.f : shadowIntensity;
      shadowIntensity = (shadowIntensity<0.f) ? 0.f : shadowIntensity;
   } 
#endif // 0
   return result;
}

/*
________________________________________________________________________________

Cylinder intersection
________________________________________________________________________________
*/
__device__ bool cylinderIntersection( 
   SceneInfo& sceneInfo,
   Primitive& cylinder,
   Material* materials, 
   char*     textures,
   int*      levels,
   Ray       ray, 
   float     timer,
   float4&   intersection,
   float4&   normal,
   float&    shadowIntensity ) 
{
   // solve the equation sphere-ray to find the intersections
   bool result = false;

   /*
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

      normal.x =  0.f;
      normal.y =  1.f;
      normal.z =  0.f;
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

      normal.x =  0.f;
      normal.y = -1.f;
      normal.z =  0.f;
   }
   */

   if( !result ) 
   {
      float4 O_C = ray.origin - cylinder.p0;
      O_C.y = 0.f;
      if(( dotProduct( O_C, ray.direction ) > 0.f ) && (vectorLength(O_C) > cylinder.p0.w)) return false;

      float a = 2.f * ( ray.direction.x*ray.direction.x + ray.direction.z*ray.direction.z );
      float b = 2.f*((ray.origin.x-cylinder.p0.x)*ray.direction.x + (ray.origin.z-cylinder.p0.z)*ray.direction.z);
      float c = O_C.x*O_C.x + O_C.z*O_C.z - cylinder.size.y*cylinder.size.y;

      float d = b*b-2.f*a*c;

      // Cylinder
      if ( /*d >= 0.f &&*/ a != 0.f) 
      {
         float r = sqrt(d);
         float t1 = (-b-r)/a;
         float t2 = (-b+r)/a;
         float ta = (t1<t2) ? t1 : t2;
         float tb = (t2<t1) ? t1 : t2;

         float4 intersection1;
         float4 intersection2;
         bool i1(false);
         bool i2(false);

         if( ta > 0.f ) 
         {
            // First intersection
            intersection1 = ray.origin+ta*ray.direction;
            intersection1.w = 0.f;
            i1 = ( fabs(intersection1.y - cylinder.p0.y) <= cylinder.size.x );
            // Transparency
            if(i1 && materials[cylinder.materialId].textureId != NO_TEXTURE ) 
            {
               float4 color = sphereMapping(cylinder, materials, textures, intersection1 );
               i1 = ((color.x+color.y+color.z) >= sceneInfo.transparentColor ); 
            }
         }

         if( tb > 0.f ) 
         {
            // Second intersection
            intersection2 = ray.origin+tb*ray.direction;
            intersection2.w = 0.f;
            i2 = ( fabs(intersection2.y - cylinder.p0.y) <= cylinder.size.x );
            if(i2 && materials[cylinder.materialId].textureId != NO_TEXTURE ) 
            {
               float4 color = sphereMapping(cylinder, materials, textures, intersection2 );
               i2 = ((color.x+color.y+color.z) >= sceneInfo.transparentColor ); 
            }
         }

         result = i1 || i2;
         if( i1 && i2 )
         {
            float4 O_I1 = intersection1 - ray.origin;
            float4 O_I2 = intersection2 - ray.origin;
            float l1 = vectorLength(O_I1);
            float l2 = vectorLength(O_I2);
            if( l1 < 0.1f ) 
            {
               intersection = intersection2;
            }
            else
            {
               if( l2 < 0.1f )
               {
                  intersection = intersection1;
               }
               else
               {
                  intersection = ( l1<l2 ) ? intersection1 : intersection2;
               }
            }
         }
         else 
         {
            intersection = i1 ? intersection1 : intersection2;
         }
      }
   }

   // Normal to surface
   if( result ) 
   {
      normal   = intersection-cylinder.p0;
      normal.y = 0.f;
      normal.w = 0.f;
      shadowIntensity = 1.f-materials[cylinder.materialId].transparency;
      if( materials[cylinder.materialId].textured ) 
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

   /*
   // Soft Shadows
   if( result && computingShadows ) 
   {
      float4 normal = normalToSurface( cylinder, intersection, depth, materials, timer ); // Normal is computed twice!!!
      normalizeVector(ray );
      normalizeVectornormal;
      shadowIntensity = 5.f*fabs(dotProduct(-ray ,normal));
      shadowIntensity = (shadowIntensity>1.f) ? 1.f : shadowIntensity;
   } 
   */
   return result;
}

/*
________________________________________________________________________________

Checkboard intersection
________________________________________________________________________________
*/
__device__ bool planeIntersection( 
   Primitive& primitive,
   Material* materials,
   char*     textures,
   int*      levels,
   Ray       ray, 
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
         float y = ray.origin.y-primitive.p0.y;
         if( reverted*ray.direction.y<0.f && reverted*ray.origin.y>reverted*primitive.p0.y) 
         {
            normal.x =  0.f;
            normal.y =  1.f;
            normal.z =  0.f;
            intersection.x = ray.origin.x+y*ray.direction.x/-ray.direction.y;
            intersection.z = ray.origin.z+y*ray.direction.z/-ray.direction.y;
            collision = 
               fabs(intersection.x - primitive.p0.x) < primitive.size.x &&
               fabs(intersection.z - primitive.p0.z) < primitive.size.y;
         }
         break;
      }
   case ptXZPlane:
      {
         float y = ray.origin.y-primitive.p0.y;
         if( reverted*ray.direction.y<0.f && reverted*ray.origin.y>reverted*primitive.p0.y) 
         {
            normal.x =  0.f;
            normal.y =  1.f;
            normal.z =  0.f;
            intersection.x = ray.origin.x+y*ray.direction.x/-ray.direction.y;
            intersection.y = primitive.p0.y;
            intersection.z = ray.origin.z+y*ray.direction.z/-ray.direction.y;
            collision = 
               fabs(intersection.x - primitive.p0.x) < primitive.size.x &&
               fabs(intersection.z - primitive.p0.z) < primitive.size.y;
         }
         if( !collision && reverted*ray.direction.y>0.f && reverted*ray.origin.y<reverted*primitive.p0.y) 
         {
            normal.x =  0.f;
            normal.y = -1.f;
            normal.z =  0.f;
            intersection.x = ray.origin.x+y*ray.direction.x/-ray.direction.y;
            intersection.y = primitive.p0.y;
            intersection.z = ray.origin.z+y*ray.direction.z/-ray.direction.y;
            collision = 
               fabs(intersection.x - primitive.p0.x) < primitive.size.x &&
               fabs(intersection.z - primitive.p0.z) < primitive.size.y;
         }
         break;
      }
   case ptYZPlane:
      {
         float x = ray.origin.x-primitive.p0.x;
         if( reverted*ray.direction.x<0.f && reverted*ray.origin.x>reverted*primitive.p0.x ) 
         {
            normal.x =  1.f;
            normal.y =  0.f;
            normal.z =  0.f;
            intersection.x = primitive.p0.x;
            intersection.y = ray.origin.y+x*ray.direction.y/-ray.direction.x;
            intersection.z = ray.origin.z+x*ray.direction.z/-ray.direction.x;
            collision = 
               fabs(intersection.y - primitive.p0.y) < primitive.size.y &&
               fabs(intersection.z - primitive.p0.z) < primitive.size.x;
         }
         if( !collision && reverted*ray.direction.x>0.f && reverted*ray.origin.x<reverted*primitive.p0.x ) 
         {
            normal.x = -1.f;
            normal.y =  0.f;
            normal.z =  0.f;
            intersection.x = primitive.p0.x;
            intersection.y = ray.origin.y+x*ray.direction.y/-ray.direction.x;
            intersection.z = ray.origin.z+x*ray.direction.z/-ray.direction.x;
            collision = 
               fabs(intersection.y - primitive.p0.y) < primitive.size.y &&
               fabs(intersection.z - primitive.p0.z) < primitive.size.x;
         }
         break;
      }
   case ptXYPlane:
      {
         float z = ray.origin.z-primitive.p0.z;
         if( reverted*ray.direction.z<0.f && reverted*ray.origin.z>reverted*primitive.p0.z) 
         {
            normal.x =  0.f;
            normal.y =  0.f;
            normal.z =  1.f;
            intersection.z = primitive.p0.z;
            intersection.x = ray.origin.x+z*ray.direction.x/-ray.direction.z;
            intersection.y = ray.origin.y+z*ray.direction.y/-ray.direction.z;
            collision = 
               fabs(intersection.x - primitive.p0.x) < primitive.size.x &&
               fabs(intersection.y - primitive.p0.y) < primitive.size.y;
         }
         if( !collision && reverted*ray.direction.z>0.f && reverted*ray.origin.z<reverted*primitive.p0.z )
         {
            normal.x =  0.f;
            normal.y =  0.f;
            normal.z = -1.f;
            intersection.z = primitive.p0.z;
            intersection.x = ray.origin.x+z*ray.direction.x/-ray.direction.z;
            intersection.y = ray.origin.y+z*ray.direction.y/-ray.direction.z;
            collision = 
               fabs(intersection.x - primitive.p0.x) < primitive.size.x &&
               fabs(intersection.y - primitive.p0.y) < primitive.size.y;
         }
         break;
      }
   case ptCamera:
      {
         if( reverted*ray.direction.z<0.f && reverted*ray.origin.z>reverted*primitive.p0.z )
         {
            normal.x =  0.f;
            normal.y =  0.f;
            normal.z =  1.f;
            intersection.z = primitive.p0.z;
            float z = ray.origin.z-primitive.p0.z;
            intersection.x = ray.origin.x+z*ray.direction.x/-ray.direction.z;
            intersection.y = ray.origin.y+z*ray.direction.y/-ray.direction.z;
            collision =
               fabs(intersection.x - primitive.p0.x) < primitive.size.x &&
               fabs(intersection.y - primitive.p0.y) < primitive.size.y;
         }
         break;
      }
   }

   if( collision ) 
   {
      shadowIntensity = 1.f;
      float4 color;
      color = materials[primitive.materialId].color;
      if(materials[primitive.materialId].textureId != NO_TEXTURE)
      {
         color = cubeMapping(primitive, materials, textures, intersection );
      }

      if( materials[primitive.materialId].transparency != 0.f && ((color.x+color.y+color.z)/3.f) >= transparentColor) 
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

#if 0
/*
________________________________________________________________________________

Triangle intersection
________________________________________________________________________________
*/
__device__ bool triangleIntersection( 
   Primitive& triangle, 
   Ray        ray,
   float      timer,
   float4&    intersection,
   float4&    normal,
   bool       computingShadows,
   float&     shadowIntensity,
   float      transparentColor
   ) 
{
   bool result = false;

   float lD = -triangle.p0.x*(triangle.p1.y*triangle.p2.z - triangle.p2.y*triangle.p1.z)
      -triangle.p1.x*(triangle.p2.y*triangle.p0.z - triangle.p0.y*triangle.p2.z)
      -triangle.p2.x*(triangle.p0.y*triangle.p1.z - triangle.p1.y*triangle.p0.z);

   float d = triangle.normal.x*ray.direction.x + triangle.normal.y*ray.direction.y + triangle.normal.z*ray.direction.z;

   d += (d==0.f) ? 0.01f : 0.f;

   float t = -(triangle.normal.x*ray.origin.x + triangle.normal.y*ray.origin.y + triangle.normal.z*ray.origin.z + lD) / d;

   if(t > 0.f)// Triangle in front of the ray
   {
      float4 i = ray.origin+t*ray.direction;

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
               normal = triangle.normal;
               result = true;
            }
         }
      }
   }
   return result;
}
#endif // 0

/*
________________________________________________________________________________

Intersection Shader
________________________________________________________________________________
*/
__device__ float4 intersectionShader( 
   SceneInfo& sceneInfo,
   Primitive& primitive, 
   Material*  materials,
   char*      textures,
#ifdef USE_KINECT
   char*      kinectVideo,
#endif // USE_KINECT
   int*       levels,
   float4     intersection,
   float      timer, 
   bool       back )
{
   float4 colorAtIntersection = materials[primitive.materialId].color;
   switch( primitive.type ) 
   {
   case ptEnvironment:
   case ptSphere:
      {
         if(materials[primitive.materialId].textureId != NO_TEXTURE)
         {
            colorAtIntersection = sphereMapping(primitive, materials, textures, intersection );
         }
         break;
      }
   case ptCheckboard :
      {
         if( materials[primitive.materialId].textureId != NO_TEXTURE ) 
         {
            colorAtIntersection = cubeMapping( primitive, materials, textures, intersection );
         }
         else 
         {
            int x = sceneInfo.viewDistance + ((intersection.x - primitive.p0.x)/primitive.p0.w*primitive.materialInfo.x);
            int z = sceneInfo.viewDistance + ((intersection.z - primitive.p0.z)/primitive.p0.w*primitive.materialInfo.y);
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
   case ptCylinder:
      {
         if(materials[primitive.materialId].textureId != NO_TEXTURE)
         {
            colorAtIntersection = sphereMapping(primitive, materials, textures, intersection );
            //colorAtIntersection = magicCylinderMapping(primitive, materials, textures, intersection, levels, timer);
         }
         break;
      }
#if 0
   case ptTriangle:
      break;
   case ptMagicCarpet:
      {
         if( materials[primitive.materialId].textureId != NO_TEXTURE ) 
         {
            colorAtIntersection = magicCarpetMapping( primitive, materials, textures, intersection, levels, timer );
         }
         break;
      }
   case ptXYPlane:
   case ptYZPlane:
   case ptXZPlane:
      {
         if( materials[primitive.materialId].textureId != NO_TEXTURE ) 
         {
            colorAtIntersection = cubeMapping( primitive, materials, textures, intersection );
         }
         break;
      }
#endif // 0
#ifdef USE_KINECT
   case ptCamera:
      {
         int x = (intersection.x-primitive.p0.x+primitive.size.x)*primitive.materialInfo.x;
         int y = gKinectVideoHeight - (intersection.y-primitive.p0.y+primitive.size.y)*primitive.materialInfo.y;

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
#endif // USE_KINECT
   }
   return colorAtIntersection;
}

/*
________________________________________________________________________________

Shadows computation
We do not consider the object from which the ray is launched...
This object cannot shadow itself !

We now have to find the intersection between the considered object and the ray 
which origin is the considered 3D float4 and which direction is defined by the 
light source center.
.
. * Lamp                     Ray = Origin -> Light Source Center
.  \
.   \##
.   #### object
.    ##
.      \
.       \  Origin
.--------O-------
.
@return 1.f when pixel is in the shades

________________________________________________________________________________
*/
__device__ float processShadows(
   SceneInfo& sceneInfo,
   BoundingBox* boudingBoxes, int nbActiveBoxes,
   Primitive* primitives,
   Material*  materials,
   char*      textures,
   int*       levels,
   int        nbPrimitives, 
   float4     lampCenter, 
   float4     origin, 
   int        objectId, 
   float      timer)
{
   float result = 0.f;
   int cptBoxes = 0;
   while( result<=sceneInfo.shadowIntensity && cptBoxes < nbActiveBoxes )
   {
      Ray ray;
      ray.origin    = origin;
      ray.direction = lampCenter-origin;
      if(boxIntersection(boudingBoxes[cptBoxes], ray, 0.f, sceneInfo.viewDistance))
      {
         BoundingBox box = boudingBoxes[cptBoxes];
         int cptPrimitives = 0;
         while( result<sceneInfo.shadowIntensity && cptPrimitives<box.nbPrimitives)
         {
            float4 intersection = {0.f,0.f,0.f,0.f};
            float4 normal       = {0.f,0.f,0.f,0.f};
            float  shadowIntensity = 0.f;

            Primitive primitive = primitives[box.primitiveIndex[cptPrimitives]];
            bool hit = false;
            switch(primitive.type)
            {
            case ptEnvironment :
            case ptSphere      : 
               hit = sphereIntersection  ( sceneInfo, primitive, materials, textures, ray, timer, intersection, normal, shadowIntensity ); 
               break;
            case ptCylinder: 
               hit = cylinderIntersection( sceneInfo, primitive, materials, textures, levels, ray, timer, intersection, normal, shadowIntensity ); 
               break;
#if 0
            case ptTriangle: 
               hit = triangleIntersection( primitive, ray, timer, intersection, normal, true, shadowIntensity, sceneInfo.transparentColor ); 
               break;
#endif // 0
            default:
               hit = planeIntersection( primitive, materials, textures, levels, ray, true, shadowIntensity, intersection, normal, sceneInfo.transparentColor, timer ); 
               if( hit ) 
               {
                  float4 O_I = intersection-origin;
                  hit = ( vectorLength(O_I)<vectorLength(ray.direction) );
               }
               break;
            }
            result = hit ? sceneInfo.shadowIntensity : 0.f;
            cptPrimitives++;
         }
      }
      cptBoxes++;
   }
   return (result>1.f) ? 1.f : result;
}

/*
________________________________________________________________________________

Primitive shader
________________________________________________________________________________
*/
__device__ float4 primitiveShader(
   SceneInfo&   sceneInfo,
   BoundingBox* boundingBoxes,
   int          nbActiveBoxes,
   Primitive* primitives,
   int        nbActivePrimitives,
   Lamp*      lamps, 
   int		  nbActiveLamps,
   Material*  materials,
   char*      textures,
#ifdef USE_KINECT
   char*      kinectVideo,
#endif // USE_KINECT
   int*       levels,
   float4     origin,
   float4     normal, 
   int        objectId, 
   float4     intersection, 
   float      timer,
   float4&    refractionFromColor,
   float&     shadowIntensity,
   float4&    totalBlinn)
{
   float4 color = materials[primitives[objectId].materialId].color * materials[primitives[objectId].materialId].specular.z;
   float4 lampsColor = { 0.f, 0.f, 0.f, 0.f };

   // Lamp Impact
   float lambert      = 0.f;
   float totalLambert = 0.f;
   shadowIntensity    = 0.f;

   if( primitives[objectId].type == ptEnvironment )
   {
      totalLambert = 1.f;
      // Final color
      color = intersectionShader( 
         sceneInfo, primitives[objectId], materials, textures, 
#ifdef USE_KINECT
         kinectVideo, 
#endif // USE_KINECT
         levels, intersection, timer, false );
   }
   else 
   {
      for( int cptLamps=0; cptLamps<nbActiveLamps; cptLamps++ ) 
      {
         if( sceneInfo.shadowsEnabled ) 
         {
            shadowIntensity = processShadows(
               sceneInfo, boundingBoxes, nbActiveBoxes,
               primitives, materials, textures, levels, 
               nbActivePrimitives, lamps[cptLamps].center, 
               intersection, objectId, timer );
         }

         float4 lightRay = lamps[cptLamps].center - intersection;
         normalizeVector(lightRay);
         // Lighted object, not in the shades
         lampsColor += lamps[cptLamps].color*lamps[cptLamps].intensity;

         // --------------------------------------------------------------------------------
         // Lambert
         // --------------------------------------------------------------------------------
         lambert = dotProduct(lightRay, normal);
         lambert = (lambert<0.f) ? 0.f : lambert;
         lambert *= (materials[primitives[objectId].materialId].refraction == 0.f) ? lamps[cptLamps].intensity : 1.f;
         lambert *= (1.f-shadowIntensity);
         totalLambert += lambert;

         if( shadowIntensity < sceneInfo.shadowIntensity )
         {
            // --------------------------------------------------------------------------------
            // Blinn - Phong
            // --------------------------------------------------------------------------------
            float4 viewRay = intersection - origin;
            normalizeVector(viewRay);

            float4 blinnDir = lightRay - viewRay;
            float temp = sqrt(dotProduct(blinnDir,blinnDir));
            if (temp != 0.f ) 
            {
               // Specular reflection
               blinnDir = (1.f / temp) * blinnDir;

               float blinnTerm = dotProduct(blinnDir,normal);
               blinnTerm = ( blinnTerm < 0.f) ? 0.f : blinnTerm;

               blinnTerm = materials[primitives[objectId].materialId].specular.x * pow(blinnTerm,materials[primitives[objectId].materialId].specular.y);
               totalBlinn += lamps[cptLamps].color * lamps[cptLamps].intensity * blinnTerm;
            }
         }
      }
      // Final color
      float4 intersectionColor = intersectionShader( 
         sceneInfo, primitives[objectId], materials, textures, 
#ifdef USE_KINECT
         kinectVideo, 
#endif // USE_KINECT
         levels, intersection, timer, false );

      color += totalLambert*intersectionColor*lampsColor;
      saturateVector(color);

      refractionFromColor = intersectionColor; // Refraction depending on color;
      saturateVector( totalBlinn );
   }

   return color;
}

/*
________________________________________________________________________________

Intersections with lamps
________________________________________________________________________________
*/
__device__ bool intersectionWithLamps( 
   Lamp* lamps, 
   int nbActiveLamps,
   Ray ray, 
   float4& lampColor)
{
   int intersections = 0; 
   lampColor.x = 0.f;
   lampColor.y = 0.f;
   lampColor.z = 0.f;

   for( int cptLamps = 0; cptLamps<nbActiveLamps; cptLamps++ ) 
   {
      float4 O_C = ray.origin - lamps[cptLamps].center; 
      float4 intersection;

      Ray r; // To do
      r.origin = ray.origin;
      r.direction = ray.direction - ray.origin;
      if( lampIntersection( lamps[cptLamps], r, O_C, intersection ) ) 
      {
         intersections++;
         lampColor += lamps[cptLamps].color;
      }
   }
   if( intersections != 0 ) 
   {
      lampColor /= intersections;
   }
   return (intersections != 0 );
}

/*
________________________________________________________________________________

Intersections with primitives
________________________________________________________________________________
*/
__device__ bool intersectionWithPrimitives(
   SceneInfo& sceneInfo,
   BoundingBox* boundingBoxes, int nbActiveBoxes,
   Primitive* primitives, int nbActivePrimitives,
   Material* materials, char* textures, int* levels,
   Ray     ray, 
   float   timer, 
   int&    closestPrimitive, 
   float4& closestIntersection,
   float4& closestNormal)
{
   bool intersections = false; 
   float minDistance  = sceneInfo.viewDistance;
   Ray r;
   r.origin    = ray.origin;
   r.direction = ray.direction - ray.origin;

   float4 intersection = {0.f,0.f,0.f,0.f};
   float4 normal       = {0.f,0.f,0.f,0.f};

   for( int cptBoxes = 0; cptBoxes < nbActiveBoxes; ++cptBoxes )
   {
      BoundingBox& box = boundingBoxes[cptBoxes];
      if( boxIntersection(box, r, 0.f, sceneInfo.viewDistance) )
      {
         if( sceneInfo.renderBoxes ) 
         {
            closestPrimitive = cptBoxes;
            return true;
         }
         int cptObjects = 0;
         //bool stop = false;
         while( /*!stop &&*/ cptObjects<box.nbPrimitives)
         { 
            bool i = false;
            float shadowIntensity = 0.f;
            Primitive& primitive = primitives[box.primitiveIndex[cptObjects]];

            float distance = vectorLength( ray.origin - primitive.p0 ) - primitive.size.x; // TODO! Not sure if i should keep it
            if( distance < minDistance )
            {
               switch( primitive.type )
               {
               case ptEnvironment :
               case ptSphere      : 
                  i = sphereIntersection  ( sceneInfo, primitive, materials, textures, r, timer, intersection, normal, shadowIntensity ); 
                  break;
               case ptCylinder: 
                  i = cylinderIntersection( sceneInfo, primitive, materials, textures, levels, r, timer, intersection, normal, shadowIntensity); 
                  break;
#if 0
               case ptTriangle: 
                  i = triangleIntersection( primitive, r, timer, intersection, normal, false, shadowIntensity, transparentColor ); 
                  break;
#endif // 0
               default        : 
                  i = planeIntersection   ( primitive, materials, textures, levels, r, false, shadowIntensity, intersection, normal, sceneInfo.transparentColor, timer); 
                  break;
               }

               if( i ) 
               {
                  float distance = vectorLength( ray.origin - intersection );
                  //stop = (cptObjects==0 && distance>minDistance);
                  if(distance>1.f && distance<minDistance) 
                  {
                     // Only keep intersection with the closest object
                     minDistance         = distance;
                     closestPrimitive    = cptObjects;
                     closestIntersection = intersection;
                     closestNormal       = normal;
                     intersections       = true;
                  } 
               }
            }
            cptObjects++;
         }
      }
   }
   return intersections;
}

/*
________________________________________________________________________________

Calculate the reflected vector                   
                                                  
                  ^ Normal to object surface (N)  
Reflection (O_R)  |                              
                \ |  Eye (O_E)                    
                 \| /                             
  ----------------O--------------- Object surface 
        closestIntersection                      
                                                   
============================================================================== 
colours                                                                                    
------------------------------------------------------------------------------ 
We now have to know the colour of this intersection                                        
Color_from_object will compute the amount of light received by the
intersection float4 and  will also compute the shadows. 
The resulted color is stored in result.                     
The first parameter is the closest object to the intersection (following 
the ray). It can  be considered as a light source if its inner light rate 
is > 0.                            
________________________________________________________________________________
*/
__device__ float4 launchRay( 
   BoundingBox* boundingBoxes, int nbActiveBoxes,
   Primitive* primitives, int nbActivePrimitives,
   Lamp*      lamps, int nbActiveLamps,
   Material*  materials,
   char* textures,
#ifdef USE_KINECT
   char*      kinectVideo, 
#endif // USE_KINECT
   int* levels,
   Ray        ray, 
   float      timer, 
   SceneInfo& sceneInfo,
   float4&    intersection,
   float&     depthOfField)
{
   float4 intersectionColor   = {0.f,0.f,0.f,0.f};
   float4 closestIntersection = {0.f,0.f,0.f,0.f};
   float4 firstIntersection   = {0.f,0.f,0.f,0.f};
   float4 normal              = {0.f,0.f,0.f,0.f};
   int    closestPrimitive;
   bool   carryon           = true;
   Ray    rayOrigin         = ray;
   float  initialRefraction = 1.0f;
   int    iteration         = 0;
   Ray    O_R = ray;
   float4 O_E;
   float4 recursiveColor[gNbIterations+1];
   float4 recursiveRatio[gNbIterations+1];
   float4 recursiveBlinn[gNbIterations+1];

   memset(recursiveColor,0,sizeof(float4)*(sceneInfo.nbRayIterations+1));
   memset(recursiveRatio,0,sizeof(float4)*(sceneInfo.nbRayIterations+1));
   memset(recursiveBlinn,0,sizeof(float4)*(sceneInfo.nbRayIterations+1));

   recursiveColor[0] = sceneInfo.backgroundColor;

   // Variable declarations
   float  shadowIntensity = 0.f;
   float4 refractionFromColor;
   float4 reflectedTarget;

   while( iteration<sceneInfo.nbRayIterations && carryon ) 
   {
      // Compute intesection with lamps
      if( intersectionWithLamps( lamps, nbActiveLamps, rayOrigin, intersectionColor ) )
      {
         recursiveColor[iteration] = intersectionColor;
         carryon = false;
      }
      else
      {
         carryon = true;
      }

      // If no intersection with lamps detected. Now compute intersection with Primitives
      if( carryon ) 
      {
         carryon = intersectionWithPrimitives(
            sceneInfo,
            boundingBoxes, nbActiveBoxes,
            primitives, nbActivePrimitives,
            materials, textures, levels,
            rayOrigin,
            timer, 
            closestPrimitive, closestIntersection, 
            normal);
      }

      if( carryon ) 
      {
         if( sceneInfo.renderBoxes ) 
         {
            recursiveColor[iteration] = materials[closestPrimitive%10].color;
         }
         else 
         {
            if ( iteration==0 )
            {
               firstIntersection = closestIntersection;
            }

            // Get object color
            recursiveColor[iteration] = primitiveShader( 
               sceneInfo,
               boundingBoxes, nbActiveBoxes,
               primitives, nbActivePrimitives, lamps, nbActiveLamps, materials, textures, 
   #ifdef USE_KINECT
               kinectVideo, 
   #endif // USE_KINECT
               levels,
               rayOrigin.origin, normal, closestPrimitive, closestIntersection, 
               timer, refractionFromColor, shadowIntensity, recursiveBlinn[iteration] );

            if( shadowIntensity != 1.f ) // No reflection/refraction if in shades
            {
               // ----------
               // Refraction
               // ----------
               if( materials[primitives[closestPrimitive].materialId].transparency != 0.f ) 
               {
                  // ----------
                  // Refraction
                  // ----------
                  // Replace the normal using the intersection color
                  // r,g,b become x,y,z... What the fuck!!
                  if( materials[primitives[closestPrimitive].materialId].textureId != NO_TEXTURE) 
                  {
                     refractionFromColor -= 0.5f;
                     normal *= refractionFromColor;
                  }

                  O_E = rayOrigin.origin - closestIntersection;
                  float refraction = materials[primitives[closestPrimitive].materialId].refraction;
                  refraction = (refraction == initialRefraction) ? 1.0f : refraction;
                  vectorRefraction( O_R.direction, O_E, refraction, normal, initialRefraction );
                  reflectedTarget = closestIntersection - O_R.direction;
                  initialRefraction = refraction;

                  recursiveRatio[iteration].x = materials[primitives[closestPrimitive].materialId].transparency;
                  recursiveRatio[iteration].z = 1.f;
               }
               else 
               {
                  // ----------
                  // Reflection
                  // ----------
                  if( materials[primitives[closestPrimitive].materialId].reflection != 0.f ) 
                  {
                     O_E = rayOrigin.origin - closestIntersection;
                     vectorReflection( O_R.direction, O_E, normal );

                     reflectedTarget = closestIntersection - O_R.direction;

                     recursiveRatio[iteration].x = materials[primitives[closestPrimitive].materialId].reflection;
                  }
                  else 
                  {
                     carryon = false;
                  }         
               }
            }
            else 
            {
               carryon = false;
            }
            rayOrigin.origin    = closestIntersection; 
            rayOrigin.direction = reflectedTarget;
         }

         iteration++; 
      }
   }

   for( int i=iteration-1; i>=0; --i ) 
   {
      recursiveColor[i] = recursiveColor[i+1]*recursiveRatio[i].x + recursiveColor[i]*(1.f-recursiveRatio[i].x);
      recursiveColor[i] += recursiveBlinn[i];
   }
   intersectionColor = recursiveColor[0];

   // Specular reflection
   //intersectionColor += recursiveBlinn[0];

   saturateVector( intersectionColor );
   intersection = closestIntersection;

   float4 O_I = firstIntersection - ray.origin;
#if 1
   // --------------------------------------------------
   // Attenation effect (Fog)
   // --------------------------------------------------
   float len = 1.f-(vectorLength(O_I)/sceneInfo.viewDistance);
   len = (len>0.f) ? len : 0.f; 
   intersectionColor.x = intersectionColor.x * len;
   intersectionColor.y = intersectionColor.y * len;
   intersectionColor.z = intersectionColor.z * len;
#endif // 0

   // Depth of field
   float4 FI_I = firstIntersection - ray.direction;
   depthOfField = (vectorLength(O_I)-depthOfField)/sceneInfo.viewDistance;
   return intersectionColor;
}


/*
________________________________________________________________________________

Main Kernel!!!
________________________________________________________________________________
*/
__global__ void k_raytracingRenderer(
   BoundingBox* BoundingBoxes, int nbActiveBoxes,
   Primitive*   primitives,    int nbActivePrimitives,
   Lamp*        lamps,         int nbActiveLamps,
   Material*    materials,
   char*        textures,
#ifdef USE_KINECT
   char*        kinectVideo,
#endif // USE_KINECT
   int*         levels,
   Ray          ray,
   float4       angles,
   SceneInfo    sceneInfo,
   float        timer,
   DepthOfFieldInfo depthOfField,
   float4*      postProcessingBuffer)
{
   int x = blockDim.x*blockIdx.x + threadIdx.x;
   int y = blockDim.y*blockIdx.y + threadIdx.y;
   int index = y*(sceneInfo.width/sceneInfo.draft)+x;

   float4 rotationCenter = {0.f,0.f,0.f,0.f};

   postProcessingBuffer[index].x = 0.f;
   postProcessingBuffer[index].y = 0.f;
   postProcessingBuffer[index].z = 0.f;
   postProcessingBuffer[index].w = 0.f;
   float dof = depthOfField.pointOfFocus;
   float4 intersection;

   if( sceneInfo.supportFor3DVision )
   {
      int halfWidth  = sceneInfo.width/2;

      Ray eyeRay;
      if( x<halfWidth ) 
      {
         // Left eye
         eyeRay.origin.x = ray.origin.x + sceneInfo.width3DVision;
         eyeRay.origin.y = ray.origin.y;
         eyeRay.origin.z = ray.origin.z;

         eyeRay.direction.x = ray.direction.x - 8.f*(float)(x - (sceneInfo.width/2) + halfWidth/2 );
         eyeRay.direction.y = ray.direction.y + 8.f*(float)(y - (sceneInfo.height/2));
         eyeRay.direction.z = ray.direction.z;
      }
      else
      {
         // Right eye
         eyeRay.origin.x = ray.origin.x - sceneInfo.width3DVision;
         eyeRay.origin.y = ray.origin.y;
         eyeRay.origin.z = ray.origin.z;

         eyeRay.direction.x = ray.direction.x - 8.f*(float)(x - (sceneInfo.width/2) - halfWidth/2);
         eyeRay.direction.y = ray.direction.y + 8.f*(float)(y - (sceneInfo.height/2));
         eyeRay.direction.z = ray.direction.z;
      }
      
      vectorRotation( eyeRay.origin, rotationCenter, angles );
      vectorRotation( eyeRay.direction, rotationCenter, angles );

      // Lamp is always behind viewer
      lamps[0].center.x = 4.f*ray.origin.x;
      lamps[0].center.y = 4.f*ray.origin.y;
      lamps[0].center.z = 2.f*ray.origin.z;

      postProcessingBuffer[index] = launchRay(
         BoundingBoxes, nbActiveBoxes,
         primitives, nbActivePrimitives,
         lamps, nbActiveLamps,
         materials,
         textures, 
   #ifdef USE_KINECT
         kinectVideo, 
   #endif // USE_KINECT
         levels,
         eyeRay, timer, 
         sceneInfo,
         intersection,
         dof);
      postProcessingBuffer[index].w = dof;
   }
   else
   {
      ray.direction.x = ray.direction.x - 8.f*(float)(sceneInfo.draft*x - (sceneInfo.width/2));
      ray.direction.y = ray.direction.y + 8.f*(float)(sceneInfo.draft*y - (sceneInfo.height/2));
      vectorRotation( ray.origin, rotationCenter, angles );
      vectorRotation( ray.direction, rotationCenter, angles );

      // Lamp is always behind viewer
      lamps[0].center.x = 4.f*ray.origin.x;
      lamps[0].center.y = 4.f*ray.origin.y;
      lamps[0].center.z = 2.f*ray.origin.z;

      postProcessingBuffer[index] = launchRay(
         BoundingBoxes, nbActiveBoxes,
         primitives, nbActivePrimitives,
         lamps, nbActiveLamps,
         materials,
         textures, 
   #ifdef USE_KINECT
         kinectVideo, 
   #endif // USE_KINECT
         levels,
         ray, timer, 
         sceneInfo,
         intersection,
         dof);
   
      postProcessingBuffer[index].w = dof;
   }
}

/*
________________________________________________________________________________

Post processing effects
________________________________________________________________________________
*/
__global__ void k_postProcessingEffects(
   SceneInfo        sceneInfo,
   DepthOfFieldInfo depthOfFieldInfo,
   float4*          postProcessingBuffer,
   float*           randoms,
   char*            bitmap) 
{
   int x = blockDim.x*blockIdx.x + threadIdx.x;
   int y = blockDim.y*blockIdx.y + threadIdx.y;
   int index = y*(sceneInfo.width/sceneInfo.draft)+x;
   if( depthOfFieldInfo.enabled )
   {
      float  depth = depthOfFieldInfo.strength*postProcessingBuffer[index].w;
      int    wh = sceneInfo.width*sceneInfo.height;

      float4 localColor;
      localColor.x = 0.f;
      localColor.y = 0.f;
      localColor.z = 0.f;

      for( int i=0; i<depthOfFieldInfo.iterations; ++i )
      {
         int ix = i%wh;
         int iy = (i+sceneInfo.width)%wh;
         int xx = x+depth*randoms[ix];
         int yy = y+depth*randoms[iy];
         if( xx>=0 && xx<sceneInfo.width && yy>=0 && yy<sceneInfo.height )
         {
            int localIndex = yy*sceneInfo.width+xx;
            if( localIndex>=0 && localIndex<wh )
            {
               localColor += postProcessingBuffer[localIndex];
            }
         }
      }
      localColor /= depthOfFieldInfo.iterations;
      localColor.w = 0.f;

      makeOpenGLColor( localColor, bitmap, index ); 
   }
   else 
   {
      makeOpenGLColor( postProcessingBuffer[index], bitmap, index ); 
   }
}

/*
________________________________________________________________________________

GPU initialization
________________________________________________________________________________
*/
extern "C" void initialize_scene( 
   int width, int height, int nbPrimitives, int nbLamps, int nbMaterials, int nbTextures, int nbLevels )
{
   // Scene resources
   checkCudaErrors(cudaMalloc( (void**)&d_boundingBoxes, NB_MAX_BOXES*sizeof(BoundingBox)));
   checkCudaErrors(cudaMalloc( (void**)&d_primitives,    nbPrimitives*sizeof(Primitive)));
   checkCudaErrors(cudaMalloc( (void**)&d_lamps,         nbLamps*sizeof(Lamp)));
   checkCudaErrors(cudaMalloc( (void**)&d_materials,     nbMaterials*sizeof(Material)));
   checkCudaErrors(cudaMalloc( (void**)&d_textures,      nbTextures*gTextureDepth*gTextureWidth*gTextureHeight));
   checkCudaErrors(cudaMalloc( (void**)&d_randoms,       width*height*sizeof(float)));
   checkCudaErrors(cudaMalloc( (void**)&d_levels,        nbLevels*sizeof(int)));

   // Rendering canvas
   checkCudaErrors(cudaMalloc( (void**)&d_postProcessingBuffer,  width*height*sizeof(float4)));
   checkCudaErrors(cudaMalloc( (void**)&d_bitmap,                width*height*gColorDepth*sizeof(char)));

#ifdef USE_KINECT
   // Kinect video and depth buffers
   checkCudaErrors(cudaMalloc( (void**)&d_kinectVideo,   gKinectVideo*gKinectVideoWidth*gKinectVideoHeight*sizeof(char)));
   checkCudaErrors(cudaMalloc( (void**)&d_kinectDepth,   gKinectDepth*gKinectDepthWidth*gKinectDepthHeight*sizeof(char)));
#endif // USE_KINECT
}

/*
________________________________________________________________________________

GPU finalization
________________________________________________________________________________
*/
extern "C" void finalize_scene()
{
   checkCudaErrors(cudaFree( d_boundingBoxes ));
   checkCudaErrors(cudaFree( d_primitives ));
   checkCudaErrors(cudaFree( d_lamps ));
   checkCudaErrors(cudaFree( d_materials ));
   checkCudaErrors(cudaFree( d_textures ));
   checkCudaErrors(cudaFree( d_randoms ));
   checkCudaErrors(cudaFree( d_levels ));
   checkCudaErrors(cudaFree( d_postProcessingBuffer ));
   checkCudaErrors(cudaFree( d_bitmap ));
#ifdef USE_KINECT
   checkCudaErrors(cudaFree( d_kinectVideo ));
   checkCudaErrors(cudaFree( d_kinectDepth ));
#endif // USE_KINECT
}

/*
________________________________________________________________________________

CPU -> GPU data transfers
________________________________________________________________________________
*/
extern "C" void h2d_scene( 
   BoundingBox* boundingBoxes, int nbActiveBoxes,
   Primitive*  primitives,     int nbPrimitives,
   Lamp*       lamps,          int nbLamps )
{
   checkCudaErrors(cudaMemcpy( d_boundingBoxes, boundingBoxes, nbActiveBoxes*sizeof(BoundingBox), cudaMemcpyHostToDevice ));
   checkCudaErrors(cudaMemcpy( d_primitives,    primitives,    nbPrimitives*sizeof(Primitive),    cudaMemcpyHostToDevice ));
   checkCudaErrors(cudaMemcpy( d_lamps,         lamps,         nbLamps*sizeof(Lamp),              cudaMemcpyHostToDevice ));
}

extern "C" void h2d_materials( 
   Material*  materials, int nbActiveMaterials,
   char*      textures , int nbActiveTextures,
   float*     randoms,   int nbRandoms,
   int*       levels,    int levelSize)
{
   checkCudaErrors(cudaMemcpy( d_materials, materials, nbActiveMaterials*sizeof(Material), cudaMemcpyHostToDevice ));
   checkCudaErrors(cudaMemcpy( d_textures,  textures,  nbActiveTextures*sizeof(char)*gTextureDepth*gTextureWidth*gTextureHeight,  cudaMemcpyHostToDevice ));
   checkCudaErrors(cudaMemcpy( d_randoms,   randoms,   nbRandoms*sizeof(float), cudaMemcpyHostToDevice ));
   checkCudaErrors(cudaMemcpy( d_levels,    levels,    levelSize*sizeof(int), cudaMemcpyHostToDevice ));
}

#ifdef USE_KINECT
extern "C" void h2d_kinect( 
   char* kinectVideo, int videoSize,
   char* kinectDepth, int depthSize )
{
   checkCudaErrors(cudaMemcpy( d_kinectVideo, kinectVideo, videoSize*sizeof(char), cudaMemcpyHostToDevice ));
   checkCudaErrors(cudaMemcpy( d_kinectDepth, kinectDepth, depthSize*sizeof(char), cudaMemcpyHostToDevice ));
}
#endif // USE_KINECT

/*
________________________________________________________________________________

GPU -> CPU data transfers
________________________________________________________________________________
*/
extern "C" void d2h_bitmap( char* bitmap, const SceneInfo sceneInfo )
{
   checkCudaErrors(cudaMemcpy( 
      bitmap, 
      d_bitmap, 
      sceneInfo.width*sceneInfo.height*gColorDepth*sizeof(char), 
      cudaMemcpyDeviceToHost ));
}

/*
________________________________________________________________________________

Kernel launcher
________________________________________________________________________________
*/
extern "C" void cudaRender(
   dim3 blockSize, int sharedMemSize,
   int nbActiveBoxes, int nbPrimitives, int nbLamps,
   Ray ray, float4 angles,
   SceneInfo sceneInfo,
   DepthOfFieldInfo depthOfFieldInfo,
   float timer)
{
   int2 size;
   size.x = static_cast<int>(sceneInfo.width/sceneInfo.draft);
   size.y = static_cast<int>(sceneInfo.height/sceneInfo.draft);
   dim3 grid((size.x+blockSize.x-1)/blockSize.x,(size.y+blockSize.y-1)/blockSize.y,1);

   k_raytracingRenderer<<<grid,blockSize,sharedMemSize>>>(
      d_boundingBoxes, nbActiveBoxes,
      d_primitives, nbPrimitives, 
      d_lamps, nbLamps,
      d_materials,
      d_textures, 
#ifdef USE_KINECT
      d_kinectVideo, 
#endif // USE_KINECT
      d_levels,
      ray, angles, 
      sceneInfo,
      timer,
      depthOfFieldInfo,
      d_postProcessingBuffer);

   k_postProcessingEffects<<<grid,blockSize>>>(
      sceneInfo, 
      depthOfFieldInfo, 
      d_postProcessingBuffer,
      d_randoms, 
      d_bitmap );
}
