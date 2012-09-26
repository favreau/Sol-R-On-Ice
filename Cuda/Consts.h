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

const int NB_MAX_PRIMITIVES = 50000; 
const int NB_MAX_BOXES      = 500;
const int NB_MAX_LAMPS      = 10;
const int NB_MAX_MATERIALS  = 100;
const int NB_MAX_TEXTURES   = 50;

const int NB_MAX_PRIMITIVES_PER_BOX = NB_MAX_PRIMITIVES/NB_MAX_BOXES;

// Constants
#define NO_MATERIAL -1
#define NO_TEXTURE  -1
#define gColorDepth    4

// Textures
#define gTextureOffset 0.f
//#define gTextureWidth  1024
//#define gTextureHeight 1024
#define gTextureWidth  256
#define gTextureHeight 256
#define gTextureDepth  3

#ifdef USE_KINECT
// Kinect
#define gKinectVideoWidth  640
#define gKinectVideoHeight 480
#define gKinectVideo       4

#define gKinectDepthWidth  320
#define gKinectDepthHeight 240
#define gKinectDepth       2
#endif // USE_KINECT