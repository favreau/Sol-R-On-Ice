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

// ________________________________________________________________________________
__device__ float vectorLength( float4 vector )
{
	return sqrt( vector.x*vector.x + vector.y*vector.y + vector.z*vector.z );
}

// ________________________________________________________________________________
#define normalizeVector( v ) \
	v /= vectorLength( v );

// ________________________________________________________________________________
__device__ void saturateVector( float4& v )
{
	v.x = (v.x>1.f) ? 1.f : v.x;
	v.y = (v.y>1.f) ? 1.f : v.y; 
	v.z = (v.z>1.f) ? 1.f : v.z;
	v.w = (v.w>1.f) ? 1.f : v.w;
}

// ________________________________________________________________________________
#define dotProduct( v1, v2 )\
	( v1.x*v2.x + v1.y*v2.y + v1.z*v2.z)

/*
________________________________________________________________________________
incident  : le vecteur normal inverse a la direction d'incidence de la source 
lumineuse
normal    : la normale a l'interface orientee dans le materiau ou se propage le 
rayon incident
reflected : le vecteur normal reflechi
________________________________________________________________________________
*/
#define vectorReflection( __r, __i, __n ) \
	__r = __i-2.f*dotProduct(__i,__n)*__n;

/*
________________________________________________________________________________
incident: le vecteur norm? inverse ? la direction d?incidence de la source 
lumineuse
n1      : index of refraction of original medium
n2      : index of refraction of new medium
________________________________________________________________________________
*/
__device__ void vectorRefraction( 
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
__device__ void vectorRotation( float4& vector, float4 center, float4 angles )
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

