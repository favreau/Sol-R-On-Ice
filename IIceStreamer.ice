#ifndef ICESTREAMER
#define ICESTREAMER

module IceStreamer
{
   // 3D vision type
   enum VisionType
   {
      vtStandard,
      vtAnaglyph,
      vt3DVision
   };

   enum OutputType
   {
      otOpenGL,
      otDelphi,
      otJPEG
   };

   // Scene information
   struct SceneInfo
   {
      int    width;
      int    height;
      int    shadowsEnabled;
      int    nbRayIterations;
      float  transparentColor;
      float  viewDistance;
      float  shadowIntensity;
      float  width3DVision;
      float  backgroundColorR;
      float  backgroundColorG;
      float  backgroundColorB;
      float  backgroundColorA;
      int    supportFor3DVision;
      int    renderBoxes;
      int    pathTracingIteration;
      int    maxPathTracingIterations;
      int    outputType; 
      int    timer;
      int    fog; // (0: disabled, 1: enabled)
      int    isometric3D;
   };

   // Post processing effect
   struct PostProcessingInfo
   {
      int   type;
      float param1; // pointOfFocus;
      float param2; // strength;
      int   param3; // iterations;
   };

   sequence<byte> bytes;

   interface BitmapProvider
   {
      bytes getBitmap(
         float ex, float ey, float ez, 
         float dx, float dy, float dz, 
         float ax, float ay, float az,
         SceneInfo scInfo,
         PostProcessingInfo ppInfo);

      SceneInfo getSceneInfo();
   };

};

#endif
