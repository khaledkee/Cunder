#ifndef CUNDER_H_
#define CUNDER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

// export and deprecated definitions
#ifdef CUNDER_STATIC_DEFINE
#define CUNDER_EXPORT
#define CUNDER_DEPRECATED
#define CUNDER_DEPRECATED_EXPORT
#else
#ifndef CUNDER_EXPORT
#if defined(_WIN32)
#ifdef CUNDER_COMPILE_LIBRARY
#define CUNDER_EXPORT __declspec(dllexport)
#else
#define CUNDER_EXPORT __declspec(dllimport)
#endif // CUNDER_COMPILE_LIBRARY
#else
#define CUNDER_EXPORT __attribute__((visibility("default")))
#endif // defined(_WIN32)
#endif // CUNDER_EXPORT
#

#ifndef CUNDER_DEPRECATED
#define CUNDER_DEPRECATED __declspec(deprecated)
#endif // CUNDER_DEPRECATED
#

#ifndef CUNDER_DEPRECATED_EXPORT
#define CUNDER_DEPRECATED_EXPORT CUNDER_EXPORT CUNDER_DEPRECATED
#endif // CUNDER_DEPRECATED_EXPORT
#endif // CUNDER_STATIC_DEFINE

// definitions
#define CUNDER_TENSOR_MAX_DIM 5

// data
typedef struct
{
	int major;
	int minor;
	int patch;
} Torch_Version;

typedef enum
{
	Cunder_Invalid,
	Cunder_Uint8,
	Cunder_Int8,
	Cunder_Int16,
	Cunder_Int32,
	Cunder_Int64,
	Cunder_Float16,
	Cunder_Float32,
	Cunder_Float64
} Cunder_DType;

struct TensorData;

typedef struct
{
	Cunder_DType dtype;
	int ndim;
	int shape[CUNDER_TENSOR_MAX_DIM];
	struct TensorData *data;
} Cunder_Tensor;

// API
CUNDER_EXPORT Torch_Version cunder_torch_version();

// Delete Tensor object.
CUNDER_EXPORT int cunder_tensor_free(Cunder_Tensor *obj);

// Initialize tensor without data
CUNDER_EXPORT Cunder_Tensor *cunder_tensor_ones(int ndim, int *shape, Cunder_DType dtype);

CUNDER_EXPORT Cunder_Tensor *cunder_tensor_zeros(int ndim, int *shape, Cunder_DType dtype);

CUNDER_EXPORT Cunder_Tensor *cunder_tensor_eye(int n, Cunder_DType dtype);

CUNDER_EXPORT Cunder_Tensor *cunder_tensor_range(int start, int end, int step, Cunder_DType dtype);

// Initialize tensor with data

CUNDER_EXPORT Cunder_Tensor *cunder_tensor_from_data_wrap(int ndim, int *shape, void *data, Cunder_DType dtype);

CUNDER_EXPORT Cunder_Tensor *cunder_tensor_from_data(int ndim, int *shape, void *data, Cunder_DType dtype, void (*free)(void *));

// Tensor to()

CUNDER_EXPORT void cunder_tensor_to(Cunder_Tensor *tensor, Cunder_DType dtype);

// Tensor toString()

CUNDER_EXPORT void cunder_tensor_print(Cunder_Tensor *tensor);

#ifdef __cplusplus
}
#endif

#endif // CUNDER_H_