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
      // Initialize Kernel
      cudaKernel_ = new CudaKernel();
      cudaKernel_->deviceQuery();
      cudaKernel_->initializeDevice( NULL, 0 );
      createRandomMaterials();

#if 0
      nbPrimitives_ = cudaKernel_->addPrimitive(ptCheckboard);
      cudaKernel_->setPrimitive( nbPrimitives_,  0, 0.f, -1000.f, 0.f, 2000, 2000, 3, 100, 100); 

      // Create Scene
      for( int i(0); i<20; ++i )
      {
         nbPrimitives_ = cudaKernel_->addPrimitive( ptSphere );
         cudaKernel_->setPrimitive( nbPrimitives_, 0,
            rand()%200-100.f, rand()%200-100.f, rand()%200-100.f, rand()%50+20.f, 0.f, rand()%nbMaterials_, 1, 1); 
      }

#else
      // PDB
      PDBReader prbReader;
      std::string fileName("./Pdb/");
      fileName += "4GME";
      fileName += ".pdb";
      prbReader.loadAtomsFromFile(fileName, *cudaKernel_, 1);
#endif // 0

      nbLamps_ = cudaKernel_->addLamp( ltSphere );
      cudaKernel_->setLamp(
         nbLamps_,
         -500.f, 500.f, -500.f, 
         0.f, 0.f, 0.f,
         0.f, 10.f, 10.f, 
         1.f, 1.f, 1.f, 1.f );

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
   for( int i(0); i<NB_MAX_MATERIALS; ++i ) 
   {
      float reflection = 0.f;
      float refraction = 0.f;
      int   texture = NO_MATERIAL;
      float transparency = 0.f;
      int   procedural = 0;
      float innerIllumination = 0.f;
      float specValue = 0.8f;
      float specPower = 100.f;
      float specCoef  = 1.f;

      float r = rand()%100/100.f;
      float g = rand()%100/100.f;
      float b = rand()%100/100.f;

      nbMaterials_ = cudaKernel_->addMaterial();
      cudaKernel_->setMaterial(
         nbMaterials_, r,  g, b,
         reflection, refraction,
         procedural, transparency, texture,
         specValue, specPower, specCoef, innerIllumination);
   }
}
