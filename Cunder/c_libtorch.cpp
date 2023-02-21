#include "c_libtorch.h"
#include <torch/all.h>
#include <torch/script.h>

namespace cunder
{
	inline static bool
	is_valid_dtype(Cunder_DType dtype)
	{
		if (dtype >= 0 && dtype < Cunder_Invalid)
		{
			return true;
		}

		return false;
	}

	// Get torch native dtype.
	constexpr c10::ScalarType
	get_libtorch_dtype(Cunder_DType dtype)
	{
		switch (dtype)
		{
		case Cunder_Bool:
			return torch::kBool;

		case Cunder_Uint8:
			return torch::kUInt8;

		case Cunder_Int8:
			return torch::kInt8;

		case Cunder_Int16:
			return torch::kInt16;

		case Cunder_Int32:
			return torch::kInt32;

		case Cunder_Int64:
			return torch::kInt64;

		case Cunder_Float32:
			return torch::kFloat32;

		case Cunder_Float64:
			return torch::kFloat64;

		case Cunder_Invalid:
		default:
			throw std::invalid_argument("Unknown dtype");
		}
	}

	// Get torch native dtype.
	constexpr Cunder_DType
	get_cunder_dtype(c10::ScalarType dtype)
	{
		switch (dtype)
		{
		case torch::kBool:
			return Cunder_Bool;

		case torch::kUInt8:
			return Cunder_Uint8;

		case torch::kInt8:
			return Cunder_Int8;

		case torch::kInt16:
			return Cunder_Int16;

		case torch::kInt32:
			return Cunder_Int32;

		case torch::kInt64:
			return Cunder_Int64;

		case torch::kFloat32:
			return Cunder_Float32;

		case torch::kFloat64:
			return Cunder_Float64;

		default:
			return Cunder_Invalid;
		}
	}

	constexpr int
	get_dtype_size(Cunder_DType dtype)
	{
		switch (dtype)
		{
		case Cunder_Bool:
			return 1;

		case Cunder_Uint8:
		case Cunder_Int8:
			return 8;

		case Cunder_Int16:
			return 16;

		case Cunder_Int32:
		case Cunder_Float32:
			return 32;

		case Cunder_Int64:
		case Cunder_Float64:
			return 64;

		case Cunder_Invalid:
		default:
			return 0;
		}
	}
} // namespace cunder

extern "C"
{
	struct Cunder_Tensor
	{
		torch::Tensor tensor;
	};

	Torch_Version
	cunder_torch_version()
	{
		return Torch_Version{TORCH_VERSION_MAJOR, TORCH_VERSION_MINOR, TORCH_VERSION_PATCH};
	}

	int
	cunder_tensor_free(Cunder_Tensor *obj)
	{
		if (obj == nullptr)
			return -1;

		free(obj);

		return 0; // success
	}

	inline static Cunder_Tensor *
	_cunder_check_initialization_param(int ndim, const int *shape, Cunder_DType dtype)
	{
		if (ndim < 1)
			return nullptr;

		if (shape == nullptr)
			return nullptr;

		if (!cunder::is_valid_dtype(dtype))
			return nullptr;

		return new Cunder_Tensor();
	}

	Cunder_Tensor *
	cunder_tensor_ones(int ndim, const int *shape, Cunder_DType dtype)
	{
		auto tensor = _cunder_check_initialization_param(ndim, shape, dtype);
		if (tensor == nullptr)
			return nullptr;

		std::vector<int64_t> vshape;
		for (int i = 0; i < ndim; i++)
			vshape.push_back(shape[i]);

		tensor->tensor = torch::ones(vshape, cunder::get_libtorch_dtype(dtype));

		return tensor;
	}

	Cunder_Tensor *
	cunder_tensor_zeros(int ndim, const int *shape, Cunder_DType dtype)
	{
		auto tensor = _cunder_check_initialization_param(ndim, shape, dtype);
		if (tensor == nullptr)
			return nullptr;

		std::vector<int64_t> vshape;
		for (int i = 0; i < ndim; i++)
			vshape.push_back(shape[i]);

		tensor->tensor = torch::zeros(vshape, cunder::get_libtorch_dtype(dtype));

		return tensor;
	}

	Cunder_Tensor *
	cunder_tensor_eye(int n, Cunder_DType dtype)
	{
		if (!cunder::is_valid_dtype(dtype))
		{
			return nullptr;
		}

		auto *tensor = new Cunder_Tensor();
		tensor->tensor = torch::eye(n, cunder::get_libtorch_dtype(dtype));
		return tensor;
	}

	Cunder_Tensor *
	cunder_tensor_range(int start, int end, int step, Cunder_DType dtype)
	{
		if (!cunder::is_valid_dtype(dtype))
		{
			return nullptr;
		}

		auto *tensor = new Cunder_Tensor();
		tensor->tensor = torch::range(start, end, step, cunder::get_libtorch_dtype(dtype));
		return tensor;
	}

	Cunder_Tensor *
	cunder_tensor_from_data_wrap(int ndim, const int *shape, void *data, Cunder_DType dtype)
	{
		auto tensor = _cunder_check_initialization_param(ndim, shape, dtype);
		if (tensor == nullptr)
			return nullptr;

		std::vector<int64_t> vshape;
		for (int i = 0; i < ndim; i++)
			vshape.push_back(shape[i]);

		tensor->tensor = torch::from_blob(data, vshape, torch::TensorOptions(cunder::get_libtorch_dtype(dtype)));
		return tensor;
	}

	Cunder_Tensor *
	cunder_tensor_from_data(int ndim, const int *shape, void *data, Cunder_DType dtype, void (*free)(void *))
	{
		auto tensor = _cunder_check_initialization_param(ndim, shape, dtype);
		if (tensor == nullptr)
			return nullptr;

		std::vector<int64_t> vshape;
		for (int i = 0; i < ndim; i++)
			vshape.push_back(shape[i]);

		tensor->tensor = torch::from_blob(data, vshape, free, torch::TensorOptions(cunder::get_libtorch_dtype(dtype)));

		return tensor;
	}

	void
	cunder_tensor_to(Cunder_Tensor *tensor, Cunder_DType dtype)
	{
		if (tensor == nullptr)
			return;
		tensor->tensor = tensor->tensor.toType(cunder::get_libtorch_dtype(dtype));
	}

	void
	cunder_tensor_print(const Cunder_Tensor *tensor)
	{
		print(tensor->tensor);
		printf("\n");
	}

	Cunder_DType
	cunder_tensor_type(const Cunder_Tensor *tensor)
	{
		return cunder::get_cunder_dtype(tensor->tensor.scalar_type());
	}

	int64_t
	cunder_tensor_ndim(const Cunder_Tensor *tensor)
	{
		return tensor->tensor.dim();
	}

	void
	cunder_tensor_shape(const Cunder_Tensor *tensor, int64_t ndim, int64_t *out_shape)
	{
		int64_t tensor_dims = tensor->tensor.dim();
		for (int64_t i = 0; i < ndim && i < tensor_dims; ++i)
			out_shape[i] = tensor->tensor.size(i);
	}

	int64_t
	cunder_tensor_numel(const Cunder_Tensor *tensor)
	{
		return tensor->tensor.numel();
	}

	int64_t
	cunder_tensor_dim_size(const Cunder_Tensor *tensor, int64_t dim)
	{
		return tensor->tensor.size(dim);
	}

	const bool *
	cunder_tensor_accessor_b(const Cunder_Tensor *tensor)
	{
		return tensor->tensor.data_ptr<bool>();
	}

	const uint8_t *
	cunder_tensor_accessor_u8(const Cunder_Tensor *tensor)
	{
		return tensor->tensor.data_ptr<uint8_t>();
	}

	const int8_t *
	cunder_tensor_accessor_i8(const Cunder_Tensor *tensor)
	{
		return tensor->tensor.data_ptr<int8_t>();
	}

	const int16_t *
	cunder_tensor_accessor_i16(const Cunder_Tensor *tensor)
	{
		return tensor->tensor.data_ptr<int16_t>();
	}

	const int32_t *
	cunder_tensor_accessor_i32(const Cunder_Tensor *tensor)
	{
		return tensor->tensor.data_ptr<int32_t>();
	}

	const int64_t *
	cunder_tensor_accessor_i64(const Cunder_Tensor *tensor)
	{
		return tensor->tensor.data_ptr<int64_t>();
	}

	const float *
	cunder_tensor_accessor_f32(const Cunder_Tensor *tensor)
	{
		return tensor->tensor.data_ptr<float>();
	}
	const double *
	cunder_tensor_accessor_f64(const Cunder_Tensor *tensor)
	{
		return tensor->tensor.data_ptr<double>();
	}

} // extern "C"
