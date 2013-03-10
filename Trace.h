#pragma once

// System
#include <windows.h>
#include <iostream>

inline std::string getTimestamp()
{
   char tmp[20];
   SYSTEMTIME time;
   GetSystemTime(&time);
   WORD millis = (time.wSecond * 1000) + time.wMilliseconds;
   sprintf_s(tmp, "%02d:%02d:%02d.%03d", time.wHour+2, time.wMinute, time.wSecond, millis/100);
   return std::string(tmp);
}

#define APPL_LOG_CONSOLE(__msg) \
   std::cout << getTimestamp() << " [" << GetCurrentThreadId() << "] " << __msg << std::endl;

#define APPL_LOG_DEBUG(__msg) \
   std::cout << getTimestamp() << " [" << GetCurrentThreadId() << "] " << __msg << std::endl;

#define APPL_LOG_INFO(__msg) \
   std::cout << getTimestamp() << " [" << GetCurrentThreadId() << "] " << __msg << std::endl;

#define APPL_LOG_WARNING(__msg) \
   std::cout << getTimestamp() << " [" << GetCurrentThreadId() << "] " << __msg << std::endl;

#define APPL_LOG_ERROR(__msg) \
   std::cout << getTimestamp() << " [" << GetCurrentThreadId() << "] " << __msg << std::endl;

#define APPL_ASSERT( stmt ) \
   if( !(stmt) ) throw cms::CMSException( "Invalid assertion: " #stmt );
