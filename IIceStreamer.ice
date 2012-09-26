#ifndef ICESTREAMER
#define ICESTREAMER

module IceStreamer
{
   // Scene information
   struct SceneInfo
   {
      int    width;
      int    height;
      float  draft;
      float  transparentColor;
      bool   shadowsEnabled;
      float  viewDistance;
      float  shadowIntensity;
      int    nbRayIterations;
      float  backgroundColorR;
      float  backgroundColorG;
      float  backgroundColorB;
      bool   supportFor3DVision;
      float  width3DVision;
      bool   renderBoxes;
   };

   // Post processing effect
   struct DepthOfFieldInfo
   {
      bool   enabled;
      float  pointOfFocus;
      float  strength;
      int    iterations;
   };

   sequence<byte> bytes;

   interface BitmapProvider
   {
      bytes getBitmap(
         float ex, float ey, float ez, 
         float dx, float dy, float dz, 
         float ax, float ay, float az,
         SceneInfo scInfo,
         DepthOfFieldInfo dofInfo);

      SceneInfo getSceneInfo();
   };

};

#endif
