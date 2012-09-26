#pragma once

#include "IIceStreamer.h"
#include "CudaKernel.h"

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
      const ::IceStreamer::SceneInfo& scInfo,
      const ::IceStreamer::DepthOfFieldInfo& dofInfo, 
      const ::Ice::Current& );

   ::IceStreamer::SceneInfo getSceneInfo(
      const ::Ice::Current& );

private:
   
   CudaKernel* cudaKernel_;
   char*       bitmap_;
   float       timer_;

};
