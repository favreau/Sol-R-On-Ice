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

const unsigned int gDraft        = 1;
const unsigned int gWindowWidth  = 511;
const unsigned int gWindowHeight = 511;
const unsigned int gWindowDepth  = 4;


IceStreamProducer::IceStreamProducer() :
   cudaKernel_(nullptr),
   producerAdapter_(nullptr),
   nbPrimitives_(0), nbLamps_(0), nbMaterials_(0), nbTextures_(20),
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
      // Initialize Kernel
      cudaKernel_ = new CudaKernel( gDraft );
      cudaKernel_->deviceQuery();
      cudaKernel_->initializeDevice( gWindowWidth, gWindowHeight, 512, 32, 20+nbTextures_, nbTextures_, NULL, 0 );
      createTextures();
      createRandomMaterials();

      // Create Scene
      for( int i(0); i<5; ++i )
      {
         nbPrimitives_ = cudaKernel_->addPrimitive( ptSphere );
         cudaKernel_->setPrimitive( nbPrimitives_,  
            rand()%800-400.f, rand()%200-100.f, rand()%800-400.f, rand()%50+20.f, 0.f, rand()%20, 1, 1); 
      }
      
      
      nbPrimitives_ = cudaKernel_->addPrimitive(ptCheckboard);
      cudaKernel_->setPrimitive( nbPrimitives_,  0.f, -100.f,    0.f, 1000, 1000, 20+rand()%nbTextures_, 2, 2); 

      nbLamps_ = cudaKernel_->addLamp( ltSphere );
      cudaKernel_->setLamp( nbLamps_, -500.f, 1000.f, -500.f, 10.f, 0.f, 1.f, 1.f, 1.f, 1.f );

      producerAdapter_ = communicator()->createObjectAdapterWithEndpoints("IIceStreamer", "tcp -p 10000 -z");
      producerAdapter_->add( new IIceStreamerImpl(cudaKernel_), communicator()->stringToIdentity("IceStreamer"));
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
   for( int i(0); i<20+nbTextures_; ++i ) 
   {
      float reflection = 0.f;
      float refraction = 0.f;
      int   texture = NO_MATERIAL;
      float transparency = 0.f;
      int   procedural = 0;
      float innerIllumination = 0.f;
      float specValue =0.8f;
      float specPower = 100.f;
      float specCoef = 1.f;

      float r = rand()%100/100.f;
      float g = rand()%100/100.f;
      float b = rand()%100/100.f;

      switch(i) 
      {
      case 0:
         r = 1.f;
         g = 1.f;
         b = 1.f;
         reflection = 0.9f;
         break;
      case 1:
         r = 1.f;
         g = 1.f;
         b = 1.f;
         reflection   = 0.1f;
         refraction   = 0.9f;
         transparency = 0.9f;
         break;
      case 2:
         r = 1.f;
         g = 1.f;
         b = 1.f;
         reflection   = 1.0f;
         refraction   = 0.5f;
         transparency = 0.9f;
         break;
      case 3:
         r = 1.f;
         g = 1.f;
         b = 1.f;
         break;
      default:
         if( i>=20 ) 
         {
            texture = i-20;
            if( texture >= 28 && texture <= 36 )
            {
               transparency = 0.5f;
               refraction   = 0.5f;
               reflection   = 0.5f;
            }
         }
         else
         {
            reflection = 0.1f;
         }
         break;
      }

      nbMaterials_ = cudaKernel_->addMaterial();
      cudaKernel_->setMaterial(
         nbMaterials_, r,  g, b,
         reflection, refraction,
         procedural, transparency, texture,
         specValue, specPower, specCoef, innerIllumination);
   }
}

void IceStreamProducer::createTextures()
{
   // Textures
   for( int i(0); i<nbTextures_; i++)
   {
      char tmp[5];
      sprintf_s(tmp, "%03d", i+1);
      std::string filename("../../../Medias/trunk/Textures/256/");
      filename += tmp;
      filename += ".bmp";
      cudaKernel_->addTexture(filename.c_str());
   }
   std::cout << nbTextures_ << " textures" << std::endl;
}
