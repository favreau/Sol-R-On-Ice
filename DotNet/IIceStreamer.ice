#ifndef ICESTREAMER
#define ICESTREAMER

module Streamer
{
   sequence<byte> bytes;

   interface BitmapProvider
   {
      void setCamera( 
         float ex, float ey, float ez, 
         float dx, float dy, float dz, 
         float ax, float ay, float az );
       bytes getBitmap(float timer, float depthOfField, float transparentColor);
   };

};

#endif
