#include "Trace.h"
#include "IIceStreamerImpl.h"

IIceStreamerImpl::IIceStreamerImpl( CudaKernel* cudaKernel ) :
   cudaKernel_(cudaKernel)
{
   SceneInfo sceneInfo = cudaKernel_->getSceneInfo();
}


IIceStreamerImpl::~IIceStreamerImpl(void)
{
}

::IceStreamer::bytes IIceStreamerImpl::getBitmap( 
   ::Ice::Float ex, ::Ice::Float ey, ::Ice::Float ez, 
   ::Ice::Float dx, ::Ice::Float dy, ::Ice::Float dz, 
   ::Ice::Float ax, ::Ice::Float ay, ::Ice::Float az,
   const ::IceStreamer::SceneInfo& scInfo,
   const ::IceStreamer::PostProcessingInfo& ppInfo, 
   const Ice::Current & )
{
	::IceStreamer::bytes result;
	try 
   {
      int colorDepth;
      switch( scInfo.outputType )
      {
      case ::IceStreamer::otOpenGL:
      case ::IceStreamer::otJPEG:
         colorDepth = 4;
      default:
         colorDepth = 3;
      }
      int imageSize = scInfo.width*scInfo.height*colorDepth;
      char* bitmap = new char[imageSize];

	   float4 eye = {ex, ey, ez, 0.f};
	   float4 direction = {dx, dy, dz, 0.f};
	   float4 angle = {ax, ay, az, 0.f};

      // Scene Information
      SceneInfo sceneInfo;
      sceneInfo.width.x              = scInfo.width;
      sceneInfo.height.x             = scInfo.height;
      sceneInfo.shadowsEnabled.x     = scInfo.shadowsEnabled;
      sceneInfo.nbRayIterations.x    = scInfo.nbRayIterations;
      sceneInfo.transparentColor.x   = scInfo.transparentColor;
      sceneInfo.viewDistance.x       = scInfo.viewDistance;
      sceneInfo.shadowIntensity.x    = scInfo.shadowIntensity;
      sceneInfo.width3DVision.x      = scInfo.width3DVision;
      sceneInfo.backgroundColor.x    = scInfo.backgroundColorR;
      sceneInfo.backgroundColor.y    = scInfo.backgroundColorG;
      sceneInfo.backgroundColor.z    = scInfo.backgroundColorB;
      sceneInfo.supportFor3DVision.x = scInfo.supportFor3DVision;
      sceneInfo.renderBoxes.x        = scInfo.renderBoxes;
      sceneInfo.pathTracingIteration.x = 0;//scInfo.pathTracingIteration;
      sceneInfo.maxPathTracingIterations.x = scInfo.maxPathTracingIterations;
      sceneInfo.misc.x               = scInfo.outputType;
      sceneInfo.misc.y               = scInfo.timer;
      sceneInfo.misc.z               = scInfo.fog;
      sceneInfo.misc.w               = scInfo.isometric3D;
      cudaKernel_->setSceneInfo(sceneInfo);

      // PostProcessing effect
      PostProcessingInfo postProcessingInfo;
      postProcessingInfo.type.x   = ppInfo.type;
      postProcessingInfo.param1.x = ppInfo.param1;
      postProcessingInfo.param2.x = ppInfo.param2;
      postProcessingInfo.param3.x = ppInfo.param3;
      cudaKernel_->setPostProcessingInfo(postProcessingInfo);

      cudaKernel_->setCamera(eye, direction, angle);
      cudaKernel_->render_begin( 0 );
      cudaKernel_->render_end( bitmap );

	   for( int i(0); i<imageSize; ++i) 
         result.push_back( bitmap[i]);

      delete [] bitmap;
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
   sceneInfo.outputType        = scInfo.misc.x;
   sceneInfo.timer             = scInfo.misc.y;
   sceneInfo.fog               = scInfo.misc.z;
   sceneInfo.isometric3D       = scInfo.misc.w;
   sceneInfo.backgroundColorR  = scInfo.backgroundColor.x;
   sceneInfo.backgroundColorG  = scInfo.backgroundColor.y;
   sceneInfo.backgroundColorB  = scInfo.backgroundColor.z;
   sceneInfo.height            = scInfo.height.x;
   sceneInfo.width             = scInfo.width.x;
   sceneInfo.nbRayIterations   = scInfo.nbRayIterations.x;
   sceneInfo.renderBoxes       = scInfo.renderBoxes.x;
   sceneInfo.shadowIntensity   = scInfo.shadowIntensity.x;
   sceneInfo.shadowsEnabled    = scInfo.shadowsEnabled.x;
   sceneInfo.supportFor3DVision= scInfo.supportFor3DVision.x;
   sceneInfo.transparentColor  = scInfo.transparentColor.x;
   sceneInfo.viewDistance      = scInfo.viewDistance.x;
   sceneInfo.width3DVision     = scInfo.width3DVision.x;
   return sceneInfo;
}
