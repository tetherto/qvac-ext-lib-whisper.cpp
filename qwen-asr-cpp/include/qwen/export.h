#pragma once

// QWEN_API marks exported symbols when building/using a shared library; empty for static.
//
// QWEN_SHARED - define when linking against or building libqwen-asr as a DLL/shared object.
// QWEN_BUILD  - define only in translation units that compile the library
//               (controls export vs import on Windows).

#ifdef QWEN_SHARED
#  if defined(_WIN32) && !defined(__MINGW32__)
#    ifdef QWEN_BUILD
#      define QWEN_API __declspec(dllexport)
#    else
#      define QWEN_API __declspec(dllimport)
#    endif
#  else
#    define QWEN_API __attribute__((visibility("default")))
#  endif
#else
#  define QWEN_API
#endif
