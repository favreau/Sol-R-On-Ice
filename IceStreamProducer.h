/*
* Copyright (C) 2010 by Lombard Odier Darier Hentsch et Cie, Geneva, Switzerland.
* ALL RIGHTS RESERVED.
*
* This software is subject to copyright protection under the laws of Switzerland
* and other countries.
*/

#pragma once

// Ice
#include <ice/ice.h>

// Project
#include "Cuda/CudaKernel.h"
#include "IIceStreamer.h"

/*
* @brief This class implements the ICE application used to produce messages
* on queues and topics held by a JMS server. This class is for testing only.
*/
class IceStreamProducer : public Ice::Application
{

public:

   // Default ctor/dtor
   IceStreamProducer();
   ~IceStreamProducer();

public:

   /**
   * @brief  Interface for an ICE object - defines a task
   * that can be run by the main thread.
   */
   virtual int run(int, char*[]);

private:

   void createRandomMaterials();

private:

   CudaKernel* cudaKernel_;

private:
   
   int nbPrimitives_;
   int nbLamps_;
   int nbMaterials_;
   int nbTextures_;

private:
   
   Ice::ObjectAdapterPtr producerAdapter_;

};
