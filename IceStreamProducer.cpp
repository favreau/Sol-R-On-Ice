/*
* Copyright (C) 2010 by Lombard Odier Darier Hentsch et Cie, Geneva, Switzerland.
* ALL RIGHTS RESERVED.
*
* This software is subject to copyright protection under the laws of Switzerland
* and other countries.
*/

// Project
#include "Trace.h"
#include "IceStreamProducer.h"
#include "IIceStreamerImpl.h"
#include "PDBReader.h"

// ------------------------------------------------------------------------------------------
// Scene
// ------------------------------------------------------------------------------------------
//float4 gBkColor = {0.5f, 0.5f, 0.5f, 0.f};
float4 gBkColor = {1.f, 1.f, 1.f, 0.f};
int    gTotalPathTracingIterations = 1;
int4   misc = {otOpenGL,0,1,1};

// Camera information
float4 gViewPos    = { 0.f, 0.f, -5000.f, 0.f };
float4 gViewDir    = { 0.f, 0.f,  3000.f, 0.f };
float4 gViewAngles = { 0.f, 0.f, 0.f, 0.f };

SceneInfo gSceneInfo = 
{ 
   768,                        // width
   512,                        // height
   true,                       // shadowsEnabled
   5,                          // nbRayIterations
   3.f,                        // transparentColor
   20000.f,                    // viewDistance
   0.9f,                       // shadowIntensity
   20.f,                       // width3DVision
   gBkColor,                   // backgroundColor
   false,                      // supportFor3DVision
   false,                      // renderBoxes
   0,                          // pathTracingIteration
   gTotalPathTracingIterations,// maxPathTracingIterations
   misc                        // outputType
};

// Scene description and behavior
int gNbBoxes      = 0;
int gNbPrimitives = 0;
int gNbLamps      = 0;
int gNbMaterials  = 0;
int gNbTextures   = 0;
float4 gRotationAngles = { 0.f, 0.f, 0.f, 0.f };
float gDefaultAtomSize(100.f);
float gDefaultStickSize(50.f);
int   gMaxPathTracingIterations = gTotalPathTracingIterations;
int   gNbMaxBoxes( 8*8*8 );
int   gGeometryType(0);
int   gAtomMaterialType(0);
float4 gRotationCenter = { 0.f, 0.f, 0.f, 0.f };

// ------------------------------------------------------------------------------------------
// Post processing
// ------------------------------------------------------------------------------------------
PostProcessingInfo gPostProcessingInfo = 
{ 
   ppe_none, 
   gViewPos.z+1000.f, 
   10.f, 
   16 
};

IceStreamProducer::IceStreamProducer() :
   cudaKernel_(nullptr),
   producerAdapter_(nullptr),
   nbPrimitives_(0), nbLamps_(0), nbMaterials_(0), nbTextures_(0),
   Ice::Application(Ice::NoSignalHandling)
{
}

IceStreamProducer::~IceStreamProducer()
{
   delete cudaKernel_;
}

int IceStreamProducer::run( int argc, char* argv[] )
{
   try
   {
      cudaKernel_ = new CudaKernel(false);
      gSceneInfo.pathTracingIteration.x = 0;
      cudaKernel_->setSceneInfo( gSceneInfo );
      cudaKernel_->initBuffers();
      cudaKernel_->setPostProcessingInfo(gPostProcessingInfo);

      createRandomMaterials();

      cudaKernel_->setCamera( gViewPos, gViewDir, gViewAngles );

      // Lamp
      int nbPrimitives = cudaKernel_->addPrimitive( ptSphere );
      cudaKernel_->setPrimitive( nbPrimitives, 50000.f, 50000.f, -50000.f, 5000.f, 0.f, 0.f, 99, 1 , 1);

      // PDB
      PDBReader prbReader;
      std::string fileName("./pdb/1BNA.pdb");
      float4 size = prbReader.loadAtomsFromFile(
         fileName, *cudaKernel_,
         static_cast<GeometryType>(gGeometryType), 
         gDefaultAtomSize, gDefaultStickSize, gAtomMaterialType, 50.f );
      gNbBoxes = cudaKernel_->compactBoxes(true);

      producerAdapter_ = communicator()->createObjectAdapter("IceStreamerAdaptor");

      IceStreamer::BitmapProviderPtr bmp = new IIceStreamerImpl(cudaKernel_);
      producerAdapter_->add( bmp, communicator()->stringToIdentity("icestreamer"));
      producerAdapter_->activate();

      communicator()->waitForShutdown();
      communicator()->destroy();
   }
   catch( const Ice::NotRegisteredException& e )
   {
      APPL_LOG_ERROR(e);
   }
   catch( const Ice::Exception& e )
   {
      APPL_LOG_ERROR(e);
   }

   return 0;
}

void IceStreamProducer::createRandomMaterials()
{
   // Materials
   for( int i(0); i<100; ++i ) 
   {
      float4 specular;
      specular.x = (i>=20 && (i/20)%2==1) ?  0.2f:   0.2f;
      specular.y = (i>=20 && (i/20)%2==1) ? 500.f: 500.0f;
      specular.z = 0.f;
      specular.w = 0.1f;

      float innerIllumination = 0.f;
      float reflection   = 0.f;

      // Transparency & refraction
      float refraction = (i>=20 && i<40) ? 1.6f : 0.f; 
      float transparency = (i>=20 && i<40) ? 0.7f : 0.f; 
      bool fastTransparency = (i>=20 && i<40);

      int   textureId = -1;
      float r,g,b;
      float noise = 0.f;
      bool  procedural = false;

      r = 0.5f+rand()%40/100.f;
      g = 0.5f+rand()%40/100.f;
      b = 0.5f+rand()%40/100.f;
      // Proteins
      switch( i%10 )
      {
      case  0: r = 0.8f;        g = 0.7f;        b = 0.7f;         break; 
      case  1: r = 0.7f;        g = 0.7f;        b = 0.7f;         break; // C Gray
      case  2: r = 174.f/255.f; g = 174.f/255.f; b = 233.f/255.f;  break; // N Blue
      case  3: r = 0.9f;        g = 0.4f;        b = 0.4f;         break; // O 
      case  4: r = 0.9f;        g = 0.9f;        b = 0.9f;         break; // H White
      case  5: r = 0.0f;        g = 0.5f;        b = 0.6f;         break; // B
      case  6: r = 0.0f;        g = 0.0f;        b = 0.7f;         break; // F
      case  7: r = 0.8f;        g = 0.6f;        b = 0.3f;         break; // P
      case  8: r = 241.f/255.f; g = 196.f/255.f; b = 107.f/255.f;  break; // S Yellow
      case  9: r = 0.9f;        g = 0.3f;        b = 0.3f;         break; // V
      }

      switch(i)
      {
         // Wall materials
      case 80: r=127.f/255.f; g=127.f/255.f; b=127.f/255.f; specular.x = 0.2f; specular.y = 10.f; specular.w = 0.3f; break;
      case 81: r=154.f/255.f; g= 94.f/255.f; b= 64.f/255.f; specular.x = 0.1f; specular.y = 100.f; specular.w = 0.1f; break;
      case 82: r= 92.f/255.f; g= 93.f/255.f; b=150.f/255.f; break; 
      ///case 83: r = 100.f/255.f; g = 20.f/255.f; b = 10.f/255.f; specular.x = 0.5f; break;
      case 83: r = 1.f; g = 1.f; b = 1.f; specular.x = 0.5f; break; // Wall color
      case 84: r = 0.8f; g = 0.8f; b = 1.f; transparency=0.85f; refraction=1.33f; /*procedural=true; */specular.x = 0.5f; specular.y = 500.f; break;

         // Lights
      case 95: r = 1.0f; g = 1.0f; b = 1.0f; refraction = 1.66f; transparency=0.9f; break;
      case 96: r = 1.0f; g = 1.0f; b = 1.0f; specular.x = 0.f; specular.y = 100.f; specular.w = 0.1f; reflection = 0.8f; break;
      case 97: r = 0.9f; g = 1.3f; b = 1.f; specular.x = 0.f; specular.y = 10.f; specular.w = 0.1f; /*textureId = 0;*/ break;
      //case 98: innerIllumination = 0.5f; break;
      case 99: r = 1.0f; g = 1.0f; b = 1.0f; innerIllumination = 1.f; break;
      }

      gNbMaterials = cudaKernel_->addMaterial();
      cudaKernel_->setMaterial( 
         gNbMaterials,
         r, g, b, noise,
         reflection, 
         refraction,
         procedural,
         false,0,
         transparency,
         textureId,
         specular.x, specular.y, specular.w, innerIllumination,
         fastTransparency );
   }
}