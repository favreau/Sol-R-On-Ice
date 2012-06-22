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

::Streamer::bytes IIceStreamerImpl::getBitmap( 
   ::Ice::Float timer, ::Ice::Float depthOfField, ::Ice::Float transparentColor, 
   const Ice::Current & )
{
   //APPL_LOG_CONSOLE("getBitmap");
   cudaKernel_->render( bitmap_, timer, depthOfField, transparentColor );
   ::Streamer::bytes result;
   
   int filesize = cudaKernel_->getImageSize();
   int w = cudaKernel_->getImageWidth();
   int h = cudaKernel_->getImageHeight();

   //Bitmap Header
   unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
   unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
   unsigned char bmppad[3] = {0,0,0};

   bmpfileheader[ 2] = (unsigned char)(filesize    );
   bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
   bmpfileheader[ 4] = (unsigned char)(filesize>>16);
   bmpfileheader[ 5] = (unsigned char)(filesize>>24);

   bmpinfoheader[ 4] = (unsigned char)(       w    );
   bmpinfoheader[ 5] = (unsigned char)(       w>> 8);
   bmpinfoheader[ 6] = (unsigned char)(       w>>16);
   bmpinfoheader[ 7] = (unsigned char)(       w>>24);
   bmpinfoheader[ 8] = (unsigned char)(       h    );
   bmpinfoheader[ 9] = (unsigned char)(       h>> 8);
   bmpinfoheader[10] = (unsigned char)(       h>>16);
   bmpinfoheader[11] = (unsigned char)(       h>>24);

   for( int i(0); i<14; ++i )
   {
      result.push_back(bmpfileheader[i]);
   }

   for( int i(0); i<40; ++i )
   {
      result.push_back(bmpinfoheader[i]);
   }

   for( int i(0); i<cudaKernel_->getImageSize(); i+=4)
   {
      result.push_back(bitmap_[i+2]);
      result.push_back(bitmap_[i+1]);
      result.push_back(bitmap_[i+0]);
      //result.push_back(bitmap_[i+3]);
   }

   result.push_back(0);
   result.push_back(0);
   result.push_back(0);
   return result;
}

void IIceStreamerImpl::setCamera(
   ::Ice::Float ex, ::Ice::Float ey, ::Ice::Float ez, 
   ::Ice::Float dx, ::Ice::Float dy, ::Ice::Float dz, 
   ::Ice::Float ax, ::Ice::Float ay, ::Ice::Float az,
   const ::Ice::Current& )
{
   //APPL_LOG_CONSOLE("setCamera");
   float4 eye = {ex, ey, ez, 0.f};
   float4 direction = {dx, dy, dz, 0.f};
   float4 angle = {ax, ay, az, 0.f};
   cudaKernel_->setCamera( eye, direction, angle );
}


std::string IIceStreamerImpl::helloWorld( 
   const std::string& something,
   const ::Ice::Current& )
{
   return something + " from Cyrille";
}
