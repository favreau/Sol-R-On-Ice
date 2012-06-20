#include "Trace.h"
#include "IIceStreamerImpl.h"

IIceStreamerImpl::IIceStreamerImpl( CudaKernel* cudaKernel ) :
   cudaKernel_(cudaKernel)
{
   bitmap_ = new unsigned char[cudaKernel_->getImageSize()];
}


IIceStreamerImpl::~IIceStreamerImpl(void)
{
   delete bitmap_;
}

::IceStreamer::bytes IIceStreamerImpl::getBitmap( 
   ::Ice::Float timer, ::Ice::Float depthOfField, ::Ice::Float transparentColor, 
   const Ice::Current & )
{
   cudaKernel_->render( bitmap_, timer, depthOfField, transparentColor );
   ::IceStreamer::bytes result;
   for( int i(0); i<cudaKernel_->getImageSize(); ++i)
   {
      result.push_back(bitmap_[i]);
   }
   return result;
}

void IIceStreamerImpl::setCamera(
   ::Ice::Float ex, ::Ice::Float ey, ::Ice::Float ez, 
   ::Ice::Float dx, ::Ice::Float dy, ::Ice::Float dz, 
   ::Ice::Float ax, ::Ice::Float ay, ::Ice::Float az,
   const ::Ice::Current& )
{
   float4 eye = {ex, ey, ez, 0.f};
   float4 direction = {dx, dy, dz, 0.f};
   float4 angle = {ax, ay, az, 0.f};
   cudaKernel_->setCamera( eye, direction, angle );
}
