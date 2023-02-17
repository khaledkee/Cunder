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

// definitions
#define CUNDER_TENSOR_MAX_DIM 5

// data
struct Torch_Version
{
	int major;
	int minor;
	int patch;
};
typedef enum
{
	Cunder_kInvalid,
	Cunder_kUint8,
	Cunder_kInt8,
	Cunder_kInt16,
	Cunder_kInt32,
	Cunder_kInt64,
	Cunder_kFloat16,
	Cunder_kFloat32,
	Cunder_kFloat64
} Cunder_DType;

struct TensorData; // Opaque

typedef struct
{
	Cunder_DType dtype;
	int ndim;
	int shape[CUNDER_TENSOR_MAX_DIM];
	struct TensorData *data;
} cunder_at_Tensor;

// API
// Delete Tensor object.
CUNDER_EXPORT int delete_cunder_at_Tensor(cunder_at_Tensor *obj);

CUNDER_EXPORT void cunder_torch_version(int *major, int *minor, int *patch);

// Returns true: CUDA is availabe, 0: CUDA is not available
CUNDER_EXPORT int cunder_cuda_is_available();

//******************************* torch::ones() *********************************//
CUNDER_EXPORT cunder_at_Tensor *cunder_torch_ones(int ndim, int *shape, Cunder_DType dtype);

// Alias for torch::ones({sz0});
CUNDER_EXPORT cunder_at_Tensor *cunder_torch_ones_1d(int sz0, Cunder_DType dtype);

// Alias for torch::ones({sz0, sz1});
CUNDER_EXPORT cunder_at_Tensor *cunder_torch_ones_2d(int sz0, int sz1, Cunder_DType dtype);

// Alias for torch::ones({sz0, sz1, sz2});
CUNDER_EXPORT cunder_at_Tensor *cunder_torch_ones_3d(int sz0, int sz1, int sz2, Cunder_DType dtype);

// Alias for torch::ones({sz0, sz1, sz2, sz3});
CUNDER_EXPORT cunder_at_Tensor *cunder_torch_ones_4d(int sz0, int sz1, int sz2, int sz3, Cunder_DType dtype);

//******************************* torch::zeros() *********************************//
CUNDER_EXPORT cunder_at_Tensor *cunder_torch_zeros(int ndim, int *shape, Cunder_DType dtype);

// Alias for torch::zeros({sz0});
CUNDER_EXPORT cunder_at_Tensor *cunder_torch_zeros_1d(int sz0, Cunder_DType dtype);

// Alias for torch::zeros({sz0, sz1});
CUNDER_EXPORT cunder_at_Tensor *cunder_torch_zeros_2d(int sz0, int sz1, Cunder_DType dtype);

// Alias for torch::zeros({sz0, sz1, sz2});
CUNDER_EXPORT cunder_at_Tensor *cunder_torch_zeros_3d(int sz0, int sz1, int sz2, Cunder_DType dtype);

// Alias for torch::zeros({sz0, sz1, sz2, sz3});
CUNDER_EXPORT cunder_at_Tensor *cunder_torch_zeros_4d(int sz0, int sz1, int sz2, int sz3, Cunder_DType dtype);

// torch::eye()
CUNDER_EXPORT cunder_at_Tensor *cunder_torch_eye(int n, Cunder_DType dtype);

#ifdef __cplusplus
}
#endif

#endif // CUNDER_H_