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

#include <cutil_inline.h>
#include <cutil_math.h>
#include "CudaDataTypes.h"

#include "Scene.cuh"
#include "Vectors.cuh"
#include "Intersections.cuh"
#include "Colors.cuh"

// Cuda Runtime paramters

#define BLOCK_SIZE 16

#if 0
// Not used so far
const int gDepthOfFieldComplexity = 1;
const int gRayCastingIterations   = 1;
const float4 gAmbientLight = { 0.4f, 0.4f, 0.4f, 0.f };
#endif // 0

// Device arrays
__constant__ __device__ Primitive* d_primitives; 
__constant__ __device__ Lamp*      d_lamps;
__constant__ __device__ Material*  d_materials;
__constant__ __device__ char*      d_textures;
__constant__ __device__ float*     d_randoms;
__constant__ __device__ int*       d_levels;
__device__   float4*    d_depthOfField;
__device__   char*      d_bitmap;

__constant__ __device__ char*      d_kinectVideo;
__constant__ __device__ char*      d_kinectDepth;

/*
* Shadows computation
* We do not consider the object from which the ray is launched...
* This object cannot shadow itself !
* 
* We now have to find the intersection between the considered object and the ray which origin is the considered 3D float4
* and which direction is defined by the light source center.
* 
* * Lamp                     Ray = Origin -> Light Source Center
*   \
*    \##
*    #### object
*     ##
*       \
*        \  Origin
* --------O-------
* 
* @return 1.f when pixel is in the shades
*/
__device__ float shadow(
   Primitive* primitives,
   Material*  materials,
   char*      textures,
   int*       levels,
   int        nbPrimitives, 
   float4     lampCenter, 
   float4     origin, 
   int        objectId, 
   float      timer,
   float      transparentColor)
{
   float4 lookAround[7] = 
   { 
      {   0.f,   0.f,   0.f,  0.f },
      { -50.f,   0.f,   0.f,  0.f },
      {   0.f,  50.f,   0.f,  0.f },
      {  50.f,   0.f,   0.f,  0.f },
      {   0.f, -50.f,   0.f,  0.f },
      {   0.f,   0.f,   1.f,  0.f },
      {   0.f,   0.f,  -1.f,  0.f } 
   };

   float result = 0.f;
   int   cptPrimitives = 0;
   int   collision = 0;
   float localShadow = 0.f;
   while( result<1.f && (collision<gNbMaxShadowCollisions) && (cptPrimitives<nbPrimitives) ) 
   {
      float shadowIntensity = 0.f;
      int hitNb = 0;
      float4 intersection = {0.f,0.f,0.f,0.f};
      float4 normal       = {0.f,0.f,0.f,0.f};
      for( int la=0; la<gSoftShadowPrecision; ++la )
      {
         float4 O_L = (lampCenter+lookAround[la])-origin;
         bool hit = false;

         switch(primitives[cptPrimitives].type)
         {
         case ptSphere  : hit = sphereIntersection  ( primitives[cptPrimitives], materials, textures, origin, O_L, timer, intersection, normal, shadowIntensity, transparentColor ); break;
         case ptCylinder: hit = cylinderIntersection( primitives[cptPrimitives], materials, textures, levels, origin, O_L, timer, intersection, normal, shadowIntensity, transparentColor ); break;
         case ptTriangle: hit = triangleIntersection( primitives[cptPrimitives], origin, O_L, timer, intersection, normal, true, shadowIntensity, transparentColor ); break;
         default:
            hit = planeIntersection( primitives[cptPrimitives], materials, textures, levels, origin, O_L, true, shadowIntensity, intersection, normal, transparentColor, timer ); 
            if( hit ) 
            {
               float4 O_I = intersection-origin;
               hit = ( vectorLength(O_I)<vectorLength(O_L) );
            }
            break;
         }

         if( hit ) 
         {
            hitNb++;
            if( primitives[cptPrimitives].type == ptSphere || primitives[cptPrimitives].type == ptCylinder || primitives[cptPrimitives].type == ptTriangle )
            {
               // Shadow exists only if object is between origin and lamp
               float4 O_I = intersection-origin;
               shadowIntensity = (vectorLength(O_I) < vectorLength(O_L)) ? shadowIntensity : 0.f;
            }
            localShadow += shadowIntensity;
         }
      }
      collision += (hitNb == 0) ? 0 : 1;
      cptPrimitives++; 
   }
   result = localShadow/gSoftShadowPrecision;
   return (result>1.f) ? 1.f : result;
}


/*
* colorFromObject 
*/
__device__ float4 colorFromObject(
   Primitive* primitives,
   int        nbActivePrimitives,
   Lamp*      lamps, 
   int		  nbActiveLamps,
   Material*  materials,
   char*      textures,
   char*      kinectVideo,
   int*       levels,
   float4     origin,
   float4     normal, 
   int        objectId, 
   float4     intersection, 
   float      timer,
   float4&    refractionFromColor,
   float&     shadowIntensity,
   float4&    totalBlinn,
   float      transparentColor)
{
   float4 color = materials[primitives[objectId].materialId].color * materials[primitives[objectId].materialId].specular.z;
   float4 lampsColor = { 0.f, 0.f, 0.f, 0.f };

   // Lamp Impact
   float lambert      = 0.f;
   float totalLambert = 0.f;
   shadowIntensity    = 0.f;

   for( int cptLamps=0; cptLamps<nbActiveLamps; cptLamps++ ) 
   {
      shadowIntensity = shadow( primitives, materials, textures, levels, nbActivePrimitives, lamps[cptLamps].center, intersection, objectId, timer, transparentColor );

      //if( (shadowIntensity) != 1.f )
      {
         // Lighted object, not in the shades
         lampsColor += lamps[cptLamps].color*lamps[cptLamps].intensity;

         // --------------------------------------------------------------------------------
         // Lambert
         // --------------------------------------------------------------------------------
         float4 lightRay = lamps[cptLamps].center - intersection;
         normalizeVector(lightRay);
         lambert = dotProduct(lightRay, normal);
         lambert = (lambert<0.f) ? 0.f : lambert;
         lambert *= (materials[primitives[objectId].materialId].refraction == 0.f) ? lamps[cptLamps].intensity : 1.f;
         lambert *= (1.f-shadowIntensity);
         totalLambert += lambert;

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
   float4 intersectionColor = objectColorAtIntersection( primitives[objectId], materials, textures, kinectVideo, levels, intersection, timer, false );

   color += totalLambert*intersectionColor*lampsColor;
   saturateVector(color);

   refractionFromColor = intersectionColor; // Refraction depending on color;
   saturateVector( totalBlinn );
   return color;
}

/**
* ________________________________________________________________________________
* Intersections with Objects
* ________________________________________________________________________________
*/
__device__ bool intersectionWithLamps( 
   Lamp* lamps, int nbActiveLamps,
   float4 origin, float4 target, 
   float4& lampColor)
{
   int intersections = 0; 
   lampColor.x = 0.f;
   lampColor.y = 0.f;
   lampColor.z = 0.f;

   for( int cptLamps = 0; cptLamps<nbActiveLamps; cptLamps++ ) 
   {
      float4 O_C = origin - lamps[cptLamps].center; 
      float4 ray = target - origin;
      float4 intersection;

      if( lampIntersection( lamps[cptLamps], origin, ray, O_C, intersection ) ) 
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

/**
* ________________________________________________________________________________
* Intersections with Objects
* ________________________________________________________________________________
*/
__device__ bool intersectionWithPrimitives( 
   Primitive* primitives, int nbActivePrimitives,
   Material* materials, char* textures, int* levels,
   float4  origin, float4 target, 
   float   timer, 
   int&    closestPrimitive, 
   float4& closestIntersection,
   float4& closestNormal,
   float   transparentColor)
{
   bool intersections = false; 
   float minDistance  = gMaxViewDistance; 
   float4 ray = target - origin; 
   float4 intersection = {0.f,0.f,0.f,0.f};
   float4 normal       = {0.f,0.f,0.f,0.f};

   for( int cptObjects = 0; cptObjects<nbActivePrimitives; cptObjects++ )
   { 
      bool i = false; 
      float shadowIntensity;

      switch( primitives[cptObjects].type )
      {
      case ptSphere  : i = sphereIntersection  ( primitives[cptObjects], materials, textures, origin, ray, timer, intersection, normal, shadowIntensity, transparentColor ); break;
      case ptCylinder: i = cylinderIntersection( primitives[cptObjects], materials, textures, levels, origin, ray, timer, intersection, normal, shadowIntensity, transparentColor); break;
      case ptTriangle: i = triangleIntersection( primitives[cptObjects], origin, ray, timer, intersection, normal, false, shadowIntensity, transparentColor ); break;
      default        : i = planeIntersection   ( primitives[cptObjects], materials, textures, levels, origin, ray, false, shadowIntensity, intersection, normal, transparentColor, timer); break;
      }

      if( i ) 
      {
         float distance = vectorLength( origin - intersection );

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
   return intersections;
}

/**
*  ------------------------------------------------------------------------------ 
* Ray Intersections
*  ============================================================================== 
*  Calculate the reflected vector                   
*                                                  
*                  ^ Normal to object surface (N)  
* Reflection (O_R)  |                              
*                 \ |  Eye (O_E)                    
*                  \| /                             
*   ----------------O--------------- Object surface 
*          closestIntersection                      
*                                                   
*  ============================================================================== 
*  colours                                                                                    
*  ------------------------------------------------------------------------------ 
*  We now have to know the colour of this intersection                                        
*  Color_from_object will compute the amount of light received by the
*  intersection float4 and  will also compute the shadows. 
*  The resulted color is stored in result.                     
*  The first parameter is the closest object to the intersection (following 
*  the ray). It can  be considered as a light source if its inner light rate 
*  is > 0.                            
*  ------------------------------------------------------------------------------ 
*/
__device__ float4 launchRay( 
   Primitive* primitives, int nbActivePrimitives,
   Lamp*      lamps, int nbActiveLamps,
   Material*  materials, char* textures,
   char*      kinectVideo, int* levels,
   float4     origin, float4 target, 
   float      timer, float transparentColor,
   float4&    intersection,
   float&     depthOfField)
{
   float4 intersectionColor   = {0.f,0.f,0.f,0.f};
   float4 closestIntersection = {0.f,0.f,0.f,0.f};
   float4 firstIntersection   = {0.f,0.f,0.f,0.f};
   float4 normal              = {0.f,0.f,0.f,0.f};
   int    closestPrimitive;
   bool   carryon           = true;
   float4 rayOrigin         = origin;
   float4 rayTarget         = target;
   float  initialRefraction = 1.0f;
   int    iteration         = 0;
   float4 O_R;
   float4 O_E;
   float4 recursiveColor[gNbIterations];
   float4 recursiveRatio[gNbIterations];
   float4 recursiveBlinn[gNbIterations];

   memset(recursiveColor,0,sizeof(float4)*gNbIterations);
   memset(recursiveRatio,0,sizeof(float4)*gNbIterations );
   memset(recursiveBlinn,0,sizeof(float4)*gNbIterations );

   // Refracted ray
   float4 reflectedOrigins[gNbIterations];
   float4 reflectedDirections[gNbIterations];
   int    reflectedRays(0);
   float4 reflectedColor = {0.f,0.f,0.f,0.f};
   float  reflectedRatio = 0.f;

   // Variable declarations
   float  shadowIntensity = 0.f;
   float4 refractionFromColor;
   float4 reflectedTarget;

   while( iteration<gNbIterations && carryon ) 
   {
      // Compute intesection with lamps
      if( intersectionWithLamps( lamps, nbActiveLamps, rayOrigin, rayTarget, intersectionColor ) )
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
            primitives, nbActivePrimitives,
            materials, textures, levels,
            rayOrigin, rayTarget,
            timer, 
            closestPrimitive, closestIntersection, 
            normal,
            transparentColor);
      }

      if( carryon ) 
      {
         if ( iteration==0 )
         {
            firstIntersection = closestIntersection;
         }

         // Get object color
         recursiveColor[iteration] = colorFromObject( 
            primitives, nbActivePrimitives, lamps, nbActiveLamps, materials, textures, kinectVideo, levels,
            origin, normal, closestPrimitive, closestIntersection, 
            timer, refractionFromColor, shadowIntensity, recursiveBlinn[iteration], transparentColor );

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
            if( materials[primitives[closestPrimitive].materialId].texture.y != NO_TEXTURE) 
            {
               refractionFromColor -= 0.5f;
               normal *= refractionFromColor;
            }

            O_E = rayOrigin - closestIntersection;
            float refraction = materials[primitives[closestPrimitive].materialId].refraction;
            refraction = (refraction == initialRefraction) ? 1.0f : refraction;
            vectorRefraction( O_R, O_E, refraction, normal, initialRefraction );
            reflectedTarget = closestIntersection - O_R;
            initialRefraction = refraction;

            recursiveRatio[iteration].x = materials[primitives[closestPrimitive].materialId].transparency;
            recursiveRatio[iteration].z = 1.f;
         }

         // ----------
         // Reflection
         // ----------
         if( reflectedRays == 0 && materials[primitives[closestPrimitive].materialId].reflection != 0.f ) 
         {
            reflectedRatio = materials[primitives[closestPrimitive].materialId].reflection;
            O_E = rayOrigin - closestIntersection;
            vectorReflection( O_R, O_E, normal );

            reflectedOrigins[reflectedRays]    = closestIntersection; 
            reflectedDirections[reflectedRays] = closestIntersection - O_R;
            reflectedRays++;
         }
         rayOrigin = closestIntersection; 
         rayTarget = reflectedTarget;

         iteration++; 
      }
   }

   for( int i(0); i<reflectedRays; ++i )
   {
      carryon = intersectionWithPrimitives(
         primitives, nbActivePrimitives,
         materials, textures, levels,
         reflectedOrigins[i], reflectedDirections[i],
         timer, 
         closestPrimitive, closestIntersection, 
         normal,
         transparentColor);
      if( carryon )
      {
         // Get object color
         reflectedColor = colorFromObject( 
            primitives, nbActivePrimitives, lamps, nbActiveLamps, materials, textures, kinectVideo, levels,
            origin, normal, closestPrimitive, closestIntersection, 
            timer, refractionFromColor, shadowIntensity, recursiveBlinn[iteration], transparentColor );
      }
   }

   for( int i=iteration-2; i>=0; --i ) 
   {
      recursiveColor[i] = (recursiveColor[i+1]*recursiveRatio[i].x + recursiveColor[i]*(1.f-recursiveRatio[i].x));
   }
   intersectionColor = recursiveColor[0]*(1.f-reflectedRatio) + reflectedColor*reflectedRatio;

   // Specular reflection
   intersectionColor += recursiveBlinn[0];

   saturateVector( intersectionColor );
   intersection = closestIntersection;

   // --------------------------------------------------
   // Attenation effect (Fog)
   // --------------------------------------------------
   float4 O_I = firstIntersection - origin;
   float len = 1.f-(vectorLength(O_I)/gMaxViewDistance);
   len = (len>0.f) ? len : 0.f; 
   intersectionColor.x = intersectionColor.x * len;
   intersectionColor.y = intersectionColor.y * len;
   intersectionColor.z = intersectionColor.z * len;

   // Depth of field
   float4 FI_I = firstIntersection - target;
   float dof = (vectorLength(FI_I)-depthOfField)/gMaxViewDistance;
   depthOfField = dof; 
   return intersectionColor;
}


/**
* ________________________________________________________________________________
* Main Kernel!!!
* ________________________________________________________________________________
*/
__global__ void render( 
   Primitive* primitives, 
   int	     nbActivePrimitives,
   Lamp*      lamps,
   int		  nbActiveLamps,
   Material*  materials,
   char*      textures,
   char*      kinectVideo,
   int*       levels,
   float4     origin,
   float4     target,
   float4     angles,
   int        width, 
   int        height, 
   float      pointOfFocus,
   int        draft,
   float      transparentColor,
   float      timer,
   float4*    depthOfField,
   char*      bitmap)
{
   int x = blockDim.x*blockIdx.x + threadIdx.x;
   int y = blockDim.y*blockIdx.y + threadIdx.y;
   int index = y*width+x;

   float4 rotationCenter = {0.f,0.f,0.f,0.f};
   vectorRotation( origin, rotationCenter, angles );

   depthOfField[index].x = 0.f;
   depthOfField[index].y = 0.f;
   depthOfField[index].z = 0.f;
   depthOfField[index].w = 0.f;

   float dof = pointOfFocus;

   target.x = target.x - 2.f*(float)(x - (width/2));
   target.y = target.y + 2.f*(float)(y - (height/2));
   vectorRotation( target, rotationCenter, angles );

   float4 intersection;
   depthOfField[index] = launchRay(
      primitives, nbActivePrimitives,
      lamps, nbActiveLamps,
      materials, textures, kinectVideo, levels,
      origin, target, timer, 
      transparentColor,
      intersection,
      dof);
   depthOfField[index].w = dof;

   //makeOpenGLColor( depthOfField[index], bitmap, index ); 
}


__global__ void postProcess(
   int width, int height,
   float4* depthOfField,
   float*  randoms,
   char*   bitmap) 
{
   int x = blockDim.x*blockIdx.x + threadIdx.x;
   int y = blockDim.y*blockIdx.y + threadIdx.y;
   int index = y*width+x;
#if 1
   float  depth = 30.f*depthOfField[index].w;
   int    wh = width*height;

   float4 localColor;
   localColor.x = 0.f;
   localColor.y = 0.f;
   localColor.z = 0.f;

   for( int i=0; i<100; ++i )
   {
      int ix = (int(i+depthOfField))%wh;
      int iy = (i+width)%wh;
      int xx = x+depth*randoms[ix];
      int yy = y+depth*randoms[iy];
      if( xx>=0 && xx<width && yy>=0 && yy<height )
      {
         int localIndex = yy*width+xx;
         if( localIndex>=0 && localIndex<wh )
         {
            localColor += depthOfField[localIndex];
         }
      }
   }
   localColor /= 100.f;
   localColor.w = 0.f;

   makeOpenGLColor( localColor, bitmap, index ); 
#else
   makeOpenGLColor( depthOfField[index], bitmap, index ); 
#endif
}

extern "C" void initialize_scene( 
   int width, int height, int nbPrimitives, int nbLamps, int nbMaterials, int nbTextures, int nbLevels )
{
   cutilSafeCall(cudaMalloc( (void**)&d_primitives,  nbPrimitives*sizeof(Primitive)));
   cutilSafeCall(cudaMalloc( (void**)&d_lamps,       nbLamps*sizeof(Lamp)));
   cutilSafeCall(cudaMalloc( (void**)&d_materials,   nbMaterials*sizeof(Material)));
   cutilSafeCall(cudaMalloc( (void**)&d_textures,    nbTextures*gTextureDepth*gTextureWidth*gTextureHeight));
   cutilSafeCall(cudaMalloc( (void**)&d_randoms,     width*height*sizeof(float)));
   cutilSafeCall(cudaMalloc( (void**)&d_levels,      nbLevels*sizeof(int)));
   cutilSafeCall(cudaMalloc( (void**)&d_depthOfField,width*height*sizeof(float4)));
   cutilSafeCall(cudaMalloc( (void**)&d_bitmap,      width*height*gColorDepth*sizeof(char)));
   cutilSafeCall(cudaMalloc( (void**)&d_kinectVideo, gKinectVideo*gKinectVideoWidth*gKinectVideoHeight*sizeof(char)));
   cutilSafeCall(cudaMalloc( (void**)&d_kinectDepth, gKinectDepth*gKinectDepthWidth*gKinectDepthHeight*sizeof(char)));
}

extern "C" void finalize_scene()
{
   cutilSafeCall(cudaFree( d_primitives ));
   cutilSafeCall(cudaFree( d_lamps ));
   cutilSafeCall(cudaFree( d_materials ));
   cutilSafeCall(cudaFree( d_textures ));
   cutilSafeCall(cudaFree( d_randoms ));
   cutilSafeCall(cudaFree( d_levels ));
   cutilSafeCall(cudaFree( d_depthOfField ));
   cutilSafeCall(cudaFree( d_bitmap ));
   cutilSafeCall(cudaFree( d_kinectVideo ));
   cutilSafeCall(cudaFree( d_kinectDepth ));
}

extern "C" void h2d_scene( 
   Primitive*  primitives, int nbPrimitives,
   Lamp*       lamps,      int nbLamps )
{
   cutilSafeCall(cudaMemcpy( d_primitives, primitives, nbPrimitives*sizeof(Primitive), cudaMemcpyHostToDevice ));
   cutilSafeCall(cudaMemcpy( d_lamps,      lamps,      nbLamps*sizeof(Lamp),           cudaMemcpyHostToDevice ));
}

extern "C" void h2d_materials( 
   Material*  materials, int nbActiveMaterials,
   char*      textures , int nbActiveTextures,
   float*     randoms,   int nbRandoms,
   int*       levels,    int levelSize)
{
   cutilSafeCall(cudaMemcpy( d_materials, materials, nbActiveMaterials*sizeof(Material), cudaMemcpyHostToDevice ));
   cutilSafeCall(cudaMemcpy( d_textures,  textures,  nbActiveTextures*sizeof(char)*gTextureDepth*gTextureWidth*gTextureHeight,  cudaMemcpyHostToDevice ));
   cutilSafeCall(cudaMemcpy( d_randoms,   randoms,   nbRandoms*sizeof(float), cudaMemcpyHostToDevice ));
   cutilSafeCall(cudaMemcpy( d_levels,    levels,    levelSize*sizeof(int), cudaMemcpyHostToDevice ));
}

extern "C" void d2h_bitmap( unsigned char* bitmap, int size )
{
   cutilSafeCall(cudaMemcpy( bitmap, d_bitmap, size, cudaMemcpyDeviceToHost ));
}

extern "C" void h2d_kinect( 
   char* kinectVideo, int videoSize,
   char* kinectDepth, int depthSize )
{
   cutilSafeCall(cudaMemcpy( d_kinectVideo, kinectVideo, videoSize*sizeof(char), cudaMemcpyHostToDevice ));
   cutilSafeCall(cudaMemcpy( d_kinectDepth, kinectDepth, depthSize*sizeof(char), cudaMemcpyHostToDevice ));
}

/**
* @brief Run the kernel on the GPU. 
* This function is executed on the Host.
*/
extern "C" void cudaRender(
   dim3 blockSize,
   int nbPrimitives, int nbLamps,
   float4 origin, float4 target, float4 angles,
   int width, int height, 
   float pointOfFocus, int draft,
   float transparentColor, float timer)
{
   // Run the Kernel
   dim3 grid((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y,1);

   render<<<grid,blockSize>>>(
      d_primitives, nbPrimitives, 
      d_lamps, nbLamps,
      d_materials, d_textures, d_kinectVideo, d_levels,
      origin, target, angles, width, height, 
      pointOfFocus, draft, 
      transparentColor, timer,
      d_depthOfField, d_bitmap );

   postProcess<<<grid,blockSize>>>(
      width, height, d_depthOfField, d_randoms, d_bitmap );
}
