// System
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include "Trace.h"
#include "IceStreamProducer.h"

int main(int argc, char* argv[]) 
{
   try
   {
      IceStreamProducer app;
      app.main( argc, argv, "IceStreamingServer.cfg" );
   }
   catch(const Ice::Exception& e)
   {
      std::cerr << e.ice_stackTrace() << std::endl;
   }
}

