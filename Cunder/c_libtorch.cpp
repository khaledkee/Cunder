#include "c_libtorch.h"
#include <torch/all.h>
#include <torch/script.h>

namespace cunder
{
	inline static bool
	is_valid_dtype(Cunder_DType dtype)
	{
		if ((dtype == Cunder_Uint8) || (dtype == Cunder_Int8) || (dtype == Cunder_Int16) || (dtype == Cunder_Int32) || (dtype == Cunder_Int64) || (dtype == Cunder_Float16) ||
			(dtype == Cunder_Float32) || (dtype == Cunder_Float64))
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
			case Cunder_Invalid:
				throw std::invalid_argument("Unknown dtype");

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

			case Cunder_Float16:
				return torch::kFloat16;

			case Cunder_Float32:
				return torch::kFloat32;

			case Cunder_Float64:
				return torch::kFloat64;
		}
	}

	constexpr int
	get_dtype_size(Cunder_DType dtype)
	{
		switch (dtype)
		{
			case Cunder_Invalid:
				return 0;

			case Cunder_Uint8:
			case Cunder_Int8:
				return 8;

			case Cunder_Int16:
			case Cunder_Float16:
				return 16;

			case Cunder_Int32:
			case Cunder_Float32:
				return 32;

			case Cunder_Int64:
			case Cunder_Float64:
				return 64;
		}
	}
} // namespace cunder

extern "C"
{
struct TensorData
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
	{
		return -1;
	}

	delete obj->data;

	free(obj);

	return 0; // success
}

inline static Cunder_Tensor *
_cunder_check_initialization_param(int ndim, const int *shape, Cunder_DType dtype)
{
	if (shape == nullptr)
	{
		return nullptr;
	}

	if (!cunder::is_valid_dtype(dtype))
	{
		return nullptr;
	}

	auto *tensor = reinterpret_cast<Cunder_Tensor *>(malloc(sizeof(Cunder_Tensor)));

	tensor->dtype = dtype;
	tensor->ndim = ndim;

	return tensor;
}

Cunder_Tensor *
cunder_tensor_ones(int ndim, int *shape, Cunder_DType dtype)
{
	auto tensor = _cunder_check_initialization_param(ndim, shape, dtype);
	if (tensor == nullptr)
		return nullptr;

	std::vector<int64_t> vshape;
	for (size_t i = 0; i < ndim; i++)
	{
		tensor->shape[i] = shape[i];
		vshape.push_back(shape[i]);
	}

	auto *data = new TensorData();
	data->tensor = torch::ones(vshape, cunder::get_libtorch_dtype(dtype));

	tensor->data = data;

	return tensor;
}

Cunder_Tensor *
cunder_tensor_zeros(int ndim, int *shape, Cunder_DType dtype)
{
	auto tensor = _cunder_check_initialization_param(ndim, shape, dtype);
	if (tensor == nullptr)
		return nullptr;

	std::vector<int64_t> vshape;
	for (size_t i = 0; i < ndim; i++)
	{
		tensor->shape[i] = shape[i];
		vshape.push_back(shape[i]);
	}

	auto *data = new TensorData();
	data->tensor = torch::zeros(vshape, cunder::get_libtorch_dtype(dtype));

	tensor->data = data;

	return tensor;
}

Cunder_Tensor *
cunder_tensor_eye(int n, Cunder_DType dtype)
{
	if (!cunder::is_valid_dtype(dtype))
	{
		return nullptr;
	}

	auto *tensor = reinterpret_cast<Cunder_Tensor *>(malloc(sizeof(Cunder_Tensor)));

	tensor->dtype = dtype;

	auto *data = new TensorData();
	data->tensor = torch::eye(n, cunder::get_libtorch_dtype(dtype));

	tensor->data = data;
	tensor->ndim = n;
	return tensor;
}

Cunder_Tensor *
cunder_tensor_range(int start, int end, int step, Cunder_DType dtype)
{
	if (!cunder::is_valid_dtype(dtype))
	{
		return nullptr;
	}

	auto *tensor = reinterpret_cast<Cunder_Tensor *>(malloc(sizeof(Cunder_Tensor)));
	tensor->dtype = dtype;
	tensor->ndim = 1;

	auto *data = new TensorData();
	data->tensor = torch::range(start, end, step, cunder::get_libtorch_dtype(dtype));

	tensor->data = data;

	return tensor;
}

Cunder_Tensor *
cunder_tensor_from_data_wrap(int ndim, int *shape, void *data, Cunder_DType dtype)
{
	auto tensor = _cunder_check_initialization_param(ndim, shape, dtype);
	if (tensor == nullptr)
		return nullptr;

	std::vector<int64_t> vshape;
	for (size_t i = 0; i < ndim; i++)
	{
		tensor->shape[i] = shape[i];
		vshape.push_back(shape[i]);
	}

	tensor->data = new TensorData();
	tensor->data->tensor = torch::from_blob(data, vshape, torch::TensorOptions(cunder::get_libtorch_dtype(dtype)));

	return tensor;
}

Cunder_Tensor *
cunder_tensor_from_data(int ndim, int *shape, void *data, Cunder_DType dtype, void (*free)(void*))
{
	auto tensor = _cunder_check_initialization_param(ndim, shape, dtype);
	if (tensor == nullptr)
		return nullptr;

	std::vector<int64_t> vshape;
	for (size_t i = 0; i < ndim; i++)
	{
		tensor->shape[i] = shape[i];
		vshape.push_back(shape[i]);
	}

	tensor->data = new TensorData();
	tensor->data->tensor = torch::from_blob(data, vshape, free, torch::TensorOptions(cunder::get_libtorch_dtype(dtype)));

	return tensor;
}

void
cunder_tensor_to(Cunder_Tensor *tensor, Cunder_DType dtype)
{
	if (tensor == nullptr)
		return;
	auto *new_data = new TensorData();
	new_data->tensor = tensor->data->tensor.toType(cunder::get_libtorch_dtype(dtype));
	delete tensor->data;
	tensor->data = new_data;
}

void
cunder_tensor_print(Cunder_Tensor *tensor)
{
	print(tensor->data->tensor);
	printf("\n");
}

} // extern "C"
