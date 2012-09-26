#pragma once

#include "IIceStreamer.h"
#include "Cuda/CudaKernel.h"

class IIceStreamerImpl : public ::IceStreamer::BitmapProvider
{

public:

   IIceStreamerImpl( CudaKernel* cudaKernel );
   ~IIceStreamerImpl(void);

public:

   ::IceStreamer::bytes getBitmap(
      ::Ice::Float ex, ::Ice::Float ey, ::Ice::Float ez, 
      ::Ice::Float dx, ::Ice::Float dy, ::Ice::Float dz, 
      ::Ice::Float ax, ::Ice::Float ay, ::Ice::Float az,
      ::Ice::Float timer, ::Ice::Float depthOfField, ::Ice::Float transparentColor, 
      const ::Ice::Current& );

private:
   
   CudaKernel*    cudaKernel_;
   unsigned char* bitmap_;

};
