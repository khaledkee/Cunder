#ifndef CUNDER_H_
#define CUNDER_H_

#include <stdint.h>
#include <stdbool.h>
#include <torch/torch.h>
#include <torch/script.h>

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

#ifndef CUNDER_DEPRECATED
#define CUNDER_DEPRECATED __declspec(deprecated)
#endif // CUNDER_DEPRECATED

#ifndef CUNDER_DEPRECATED_EXPORT
#define CUNDER_DEPRECATED_EXPORT CUNDER_EXPORT CUNDER_DEPRECATED
#endif // CUNDER_DEPRECATED_EXPORT
#endif // CUNDER_STATIC_DEFINE

	// definitions
	struct Cunder_Tensor
	{
		torch::Tensor tensor;
	};

	struct Cunder_Module
	{
		torch::jit::Module module;
	};

	// data
	typedef struct
	{
		int major;
		int minor;
		int patch;
	} Torch_Version;

	typedef enum
	{
		Cunder_Bool,
		Cunder_Uint8,
		Cunder_Int8,
		Cunder_Int16,
		Cunder_Int32,
		Cunder_Int64,
		Cunder_Float32,
		Cunder_Float64,
		Cunder_Invalid
	} Cunder_DType;

	typedef struct Cunder_Tensor Cunder_Tensor;
	typedef struct Cunder_Module Cunder_Module;

	// API
	CUNDER_EXPORT Torch_Version
	cunder_torch_version();

	// Delete objects.
	CUNDER_EXPORT int
	cunder_tensor_free(Cunder_Tensor *self);

	CUNDER_EXPORT Cunder_Tensor *
	cunder_tensor_clone(Cunder_Tensor *src);

	CUNDER_EXPORT Cunder_Tensor *
	allocated_cunder_tensor_clone(Cunder_Tensor *out, Cunder_Tensor *src);

	CUNDER_EXPORT int
	cunder_module_free(Cunder_Module *self);

	// Initialize tensor without data
	CUNDER_EXPORT Cunder_Tensor *
	cunder_tensor_ones(int ndim, const int *shape, Cunder_DType dtype);

	CUNDER_EXPORT Cunder_Tensor *
	allocated_cunder_tensor_ones(Cunder_Tensor *tensor, int ndim, const int *shape, Cunder_DType dtype);

	CUNDER_EXPORT Cunder_Tensor *
	cunder_tensor_zeros(int ndim, const int *shape, Cunder_DType dtype);

	CUNDER_EXPORT Cunder_Tensor *
	allocated_cunder_tensor_zeros(Cunder_Tensor *tensor, int ndim, const int *shape, Cunder_DType dtype);

	CUNDER_EXPORT Cunder_Tensor *
	cunder_tensor_eye(int n, Cunder_DType dtype);

	CUNDER_EXPORT Cunder_Tensor *
	allocated_cunder_tensor_eye(Cunder_Tensor *tensor, int n, Cunder_DType dtype);

	CUNDER_EXPORT Cunder_Tensor *
	cunder_tensor_range(int start, int end, int step, Cunder_DType dtype);

	CUNDER_EXPORT Cunder_Tensor *
	allocated_cunder_tensor_range(Cunder_Tensor *tensor, int start, int end, int step, Cunder_DType dtype);

	// Initialize tensor with data

	CUNDER_EXPORT Cunder_Tensor *
	cunder_tensor_from_data_wrap(int ndim, const int *shape, void *data, Cunder_DType dtype);

	CUNDER_EXPORT Cunder_Tensor *
	allocated_cunder_tensor_from_data_wrap(Cunder_Tensor *tensor, int ndim, const int *shape, void *data, Cunder_DType dtype);

	CUNDER_EXPORT Cunder_Tensor *
	cunder_tensor_from_data(int ndim, const int *shape, void *data, Cunder_DType dtype, void (*free)(void *));

	CUNDER_EXPORT Cunder_Tensor *
	allocated_cunder_tensor_from_data(
		Cunder_Tensor *tensor, int ndim, const int *shape, void *data, Cunder_DType dtype, void (*free)(void *));

	// Tensor to()

	CUNDER_EXPORT void
	cunder_tensor_to(Cunder_Tensor *tensor, Cunder_DType dtype);

	// Tensor print()

	CUNDER_EXPORT void
	cunder_tensor_print(const Cunder_Tensor *tensor);

	// Tensor info

	CUNDER_EXPORT Cunder_DType
	cunder_tensor_type(const Cunder_Tensor *tensor);
	CUNDER_EXPORT int64_t
	cunder_tensor_ndim(const Cunder_Tensor *tensor);
	CUNDER_EXPORT int64_t *
	cunder_tensor_shape(const Cunder_Tensor *tensor, int64_t &ndim);
	CUNDER_EXPORT int64_t
	cunder_tensor_numel(const Cunder_Tensor *tensor);
	CUNDER_EXPORT int64_t
	cunder_tensor_dim_size(const Cunder_Tensor *tensor, int64_t dim);

	// tensor accessors

	CUNDER_EXPORT const bool *
	cunder_tensor_accessor_b(const Cunder_Tensor *tensor);
	CUNDER_EXPORT const uint8_t *
	cunder_tensor_accessor_u8(const Cunder_Tensor *tensor);
	CUNDER_EXPORT const int8_t *
	cunder_tensor_accessor_i8(const Cunder_Tensor *tensor);
	CUNDER_EXPORT const int16_t *
	cunder_tensor_accessor_i16(const Cunder_Tensor *tensor);
	CUNDER_EXPORT const int32_t *
	cunder_tensor_accessor_i32(const Cunder_Tensor *tensor);
	CUNDER_EXPORT const int64_t *
	cunder_tensor_accessor_i64(const Cunder_Tensor *tensor);
	CUNDER_EXPORT const float *
	cunder_tensor_accessor_f32(const Cunder_Tensor *tensor);
	CUNDER_EXPORT const double *
	cunder_tensor_accessor_f64(const Cunder_Tensor *tensor);

	// torch jit load
	CUNDER_EXPORT Cunder_Module *
	cunder_module_load(const char *filename);

	CUNDER_EXPORT void
	cunder_module_dump(
		const Cunder_Module *module, bool print_method_bodies = false, bool print_attr_values = false, bool print_param_values = false);

	CUNDER_EXPORT void
	cunder_print_torch_attrs(Cunder_Tensor *tensor);

#ifdef __cplusplus
}
#endif

#endif // CUNDER_H_