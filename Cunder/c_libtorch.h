#ifndef CUNDER_H_
#define CUNDER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// export and deprecated definitions
#ifdef CUNDER_STATIC_DEFINE
#  define CUNDER_EXPORT
#  define CUNDER_DEPRECATED
#  define CUNDER_DEPRECATED_EXPORT
#else
#  ifndef CUNDER_EXPORT
#    if defined(_WIN32)
#      ifdef CUNDER_COMPILE_LIBRARY
#        define CUNDER_EXPORT __declspec(dllexport)
#      else
#        define CUNDER_EXPORT __declspec(dllimport)
#      endif // CUNDER_COMPILE_LIBRARY
#    else
#      define CUNDER_EXPORT __attribute__((visibility("default")))
#    endif // defined(_WIN32)
#  endif // CUNDER_EXPORT
#
#  ifndef CUNDER_DEPRECATED
#    define CUNDER_DEPRECATED __declspec(deprecated)
#  endif // CUNDER_DEPRECATED
#
#  ifndef CUNDER_DEPRECATED_EXPORT
#    define CUNDER_DEPRECATED_EXPORT CUNDER_EXPORT CUNDER_DEPRECATED
#  endif // CUNDER_DEPRECATED_EXPORT
#endif // CUNDER_STATIC_DEFINE



// library start

// definitions
#define CUNDER_TENSOR_MAX_DIM 5

// data
struct Torch_Version
{
	int major;
	int minor;
	int patch;
};

// API
CUNDER_EXPORT Torch_Version
c_torch_version();

#ifdef __cplusplus
}
#endif

#endif // CUNDER_H_