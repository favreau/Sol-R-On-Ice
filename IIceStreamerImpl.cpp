#include "Trace.h"
#include "IIceStreamerImpl.h"

IIceStreamerImpl::IIceStreamerImpl( CudaKernel* cudaKernel ) :
   cudaKernel_(cudaKernel)
{
   SceneInfo sceneInfo = cudaKernel_->getSceneInfo();
   int imageSize = sceneInfo.width*sceneInfo.height*gColorDepth;
   bitmap_ = new char[imageSize];
   timer_ = 0;
}


IIceStreamerImpl::~IIceStreamerImpl(void)
{
   delete bitmap_;
}

::IceStreamer::bytes IIceStreamerImpl::getBitmap( 
   ::Ice::Float ex, ::Ice::Float ey, ::Ice::Float ez, 
   ::Ice::Float dx, ::Ice::Float dy, ::Ice::Float dz, 
   ::Ice::Float ax, ::Ice::Float ay, ::Ice::Float az,
   const ::IceStreamer::SceneInfo& scInfo,
   const ::IceStreamer::DepthOfFieldInfo& dofInfo, 
   const Ice::Current & )
{
	::IceStreamer::bytes result;
	try {
	   float4 eye = {ex, ey, ez, 0.f};
	   float4 direction = {dx, dy, dz, 0.f};
	   float4 angle = {ax, ay, az, 0.f};

      // Scene Information
      SceneInfo sceneInfo;
      sceneInfo.backgroundColor.x = scInfo.backgroundColorR;
      sceneInfo.backgroundColor.y = scInfo.backgroundColorG;
      sceneInfo.backgroundColor.z = scInfo.backgroundColorB;
      sceneInfo.draft             = scInfo.draft;
      sceneInfo.height            = scInfo.height;
      sceneInfo.width             = scInfo.width;
      sceneInfo.nbRayIterations   = scInfo.nbRayIterations;
      sceneInfo.renderBoxes       = scInfo.renderBoxes;
      sceneInfo.shadowIntensity   = scInfo.shadowIntensity;
      sceneInfo.shadowsEnabled    = scInfo.shadowsEnabled;
      sceneInfo.supportFor3DVision= scInfo.supportFor3DVision;
      sceneInfo.transparentColor  = scInfo.transparentColor;
      sceneInfo.viewDistance      = scInfo.viewDistance;
      sceneInfo.width3DVision     = scInfo.width3DVision;
      cudaKernel_->setSceneInfo(sceneInfo);

      // Depth of field effect
      DepthOfFieldInfo deathOfFieldInfo;
      deathOfFieldInfo.enabled      = dofInfo.enabled;
      deathOfFieldInfo.iterations   = dofInfo.iterations;
      deathOfFieldInfo.pointOfFocus = dofInfo.pointOfFocus;
      deathOfFieldInfo.strength     = dofInfo.strength;
      cudaKernel_->setDepthOfFieldInfo(deathOfFieldInfo);

      cudaKernel_->setCamera(eye, direction, angle);
      cudaKernel_->render( bitmap_, timer_ );

      int imageSize = sceneInfo.width*sceneInfo.height*gColorDepth;
	   for( int i(0); i<imageSize; ++i)
	   {
		  result.push_back(bitmap_[i]);
	   }
	}
	catch( ... )
	{
		std::cout << "*** ERROR *** getBitmap failed" << std::endl;
	}
   return result;
}

::IceStreamer::SceneInfo IIceStreamerImpl::getSceneInfo(
  const ::Ice::Current& )
{
   // Scene Information
   SceneInfo scInfo = cudaKernel_->getSceneInfo();
   ::IceStreamer::SceneInfo sceneInfo;
   sceneInfo.backgroundColorR  = scInfo.backgroundColor.x;
   sceneInfo.backgroundColorG  = scInfo.backgroundColor.y;
   sceneInfo.backgroundColorB  = scInfo.backgroundColor.z;
   sceneInfo.draft             = scInfo.draft;
   sceneInfo.height            = scInfo.height;
   sceneInfo.width             = scInfo.width;
   sceneInfo.nbRayIterations   = scInfo.nbRayIterations;
   sceneInfo.renderBoxes       = scInfo.renderBoxes;
   sceneInfo.shadowIntensity   = scInfo.shadowIntensity;
   sceneInfo.shadowsEnabled    = scInfo.shadowsEnabled;
   sceneInfo.supportFor3DVision= scInfo.supportFor3DVision;
   sceneInfo.transparentColor  = scInfo.transparentColor;
   sceneInfo.viewDistance      = scInfo.viewDistance;
   sceneInfo.width3DVision     = scInfo.width3DVision;
   return sceneInfo;
}