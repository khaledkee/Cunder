#ifndef CUNDER_H_
#define CUNDER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// export and deprecated definitions
#ifdef CUNDER_STATIC_DEFINE
#  define CUNDER_EXPORT
#else
#ifndef CUNDER_EXPORT
#  if defined(_WIN32)
#    ifdef CUNDER_COMPILE_LIBRARY
#      define CUNDER_EXPORT __declspec(dllexport)
#    else
#      define CUNDER_EXPORT __declspec(dllimport)
#    endif // CUNDER_COMPILE_LIBRARY
#  else
#    define CUNDER_EXPORT __attribute__((visibility("default")))
#  endif // defined(_WIN32)
#endif // CUNDER_EXPORT

#ifndef CUNDER_DEPRECATED
#  define CUNDER_DEPRECATED __declspec(deprecated)
#endif // CUNDER_DEPRECATED

#ifndef CUNDER_DEPRECATED_EXPORT
#  define CUNDER_DEPRECATED_EXPORT CUNDER_EXPORT CUNDER_DEPRECATED
#endif // CUNDER_DEPRECATED_EXPORT

#define C_TORCH_TENSOR_MAX_DIM  (8)

#define C_TORCH_DEFAULT_DIM (-1)
#define C_TORCH_DEFAULT_N (-1)

#ifdef __cplusplus
}
#endif

// library start
#define CUNDER_TENSOR_MAX_DIM 5


#endif // CUNDER_H_