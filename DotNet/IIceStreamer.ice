#ifndef ICESTREAMER
#define ICESTREAMER

module Streamer
{
   sequence<byte> bytes;

   interface BitmapProvider
   {
      bytes getBitmap(
         float ex, float ey, float ez, 
         float dx, float dy, float dz, 
         float ax, float ay, float az,
         float timer, float depthOfField, float transparentColor);

      string helloWorld( string something );
   };

};

#endif
