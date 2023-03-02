#include <torch/all.h>
#include <torch/script.h>
#include "c_libtorch.h"

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

	// definitions
	struct Cunder_Tensor
	{
		torch::Tensor tensor;
	};

	struct Cunder_Module
	{
		torch::jit::Module module;
	};

	void
	cunder_get_tensor_size_alignment(int64_t &size, int64_t &alignment)
	{
		size = sizeof(Cunder_Tensor);
		alignment = alignof(Cunder_Tensor);
	}

	void
	cunder_get_module_size_alignment(int64_t &size, int64_t &alignment)
	{
		size = sizeof(Cunder_Module);
		alignment = alignof(Cunder_Module);
	}

	Torch_Version
	cunder_torch_version()
	{
		return Torch_Version{TORCH_VERSION_MAJOR, TORCH_VERSION_MINOR, TORCH_VERSION_PATCH};
	}

	Cunder_Tensor *
	cunder_tensor_clone(Cunder_Tensor *src)
	{
		if (src == nullptr)
			return nullptr;

		auto out = new Cunder_Tensor{};
		out->tensor = src->tensor.clone();
		return out;
	}

	void
	cunder_tensor_clone_allocated(void *out_void, void *src_void)
	{
		Cunder_Tensor *src = (Cunder_Tensor *)src_void;
		Cunder_Tensor *out = (Cunder_Tensor *)out_void;
		if (src == nullptr)
			return;

		if (out->tensor.getIntrusivePtr() == nullptr)
			out->tensor.unsafeReleaseTensorImpl();
		out->tensor = src->tensor.clone();
	}

	int
	cunder_tensor_free(Cunder_Tensor *self)
	{
		if (self == nullptr)
			return -1;

		self->tensor.~Tensor();
		free(self);

		return 0; // success
	}

	int
	cunder_module_free(Cunder_Module *self)
	{
		if (self == nullptr)
			return -1;

		self->module.~Module();
		free(self);

		return 0; // success
	}

	int
	cunder_tensor_free_allocated(void *self_void)
	{
		Cunder_Tensor *self = (Cunder_Tensor *)self_void;
		if (self == nullptr)
			return -1;

		self->tensor.~Tensor();
		_aligned_free(self);

		return 0; // success
	}

	int
	cunder_module_free_allocated(void *self_void)
	{
		Cunder_Module *self = (Cunder_Module *)self_void;
		if (self == nullptr)
			return -1;

		self->module.~Module();
		_aligned_free(self);

		return 0; // success
	}

	inline static bool
	_cunder_check_initialization_param(int ndim, const int *shape, Cunder_DType dtype)
	{
		if (ndim < 1 || ndim > 3 || shape == nullptr || cunder::is_valid_dtype(dtype) == false)
			return false;
		return true;
	}

	Cunder_Tensor *
	cunder_tensor_ones(int ndim, const int *shape, Cunder_DType dtype)
	{
		Cunder_Tensor *tensor;
		if (_cunder_check_initialization_param(ndim, shape, dtype))
			tensor = new Cunder_Tensor{};
		else
			return nullptr;

		std::vector<int64_t> vshape(shape, shape + ndim);
		tensor->tensor = torch::ones(vshape, cunder::get_libtorch_dtype(dtype));
		return tensor;
	}

	Cunder_Tensor *
	cunder_tensor_zeros(int ndim, const int *shape, Cunder_DType dtype)
	{
		Cunder_Tensor *tensor;
		if (_cunder_check_initialization_param(ndim, shape, dtype))
			tensor = new Cunder_Tensor{};
		else
			return nullptr;

		std::vector<int64_t> vshape(shape, shape + ndim);
		tensor->tensor = torch::zeros(vshape, cunder::get_libtorch_dtype(dtype));
		return tensor;
	}

	Cunder_Tensor *
	cunder_tensor_eye(int n, Cunder_DType dtype)
	{
		if (cunder::is_valid_dtype(dtype) == false)
			return nullptr;

		auto *tensor = new Cunder_Tensor();
		tensor->tensor = torch::eye(n, cunder::get_libtorch_dtype(dtype));
		return tensor;
	}

	Cunder_Tensor *
	cunder_tensor_range(int start, int end, int step, Cunder_DType dtype)
	{
		if (cunder::is_valid_dtype(dtype) == false)
			return nullptr;

		auto *tensor = new Cunder_Tensor();
		tensor->tensor = torch::range(start, end, step, cunder::get_libtorch_dtype(dtype));
		return tensor;
	}

	void
	cunder_tensor_ones_allocated(void *tensor_void, int ndim, const int *shape, Cunder_DType dtype)
	{
		if (_cunder_check_initialization_param(ndim, shape, dtype) == false)
			return;

		std::vector<int64_t> vshape(shape, shape + ndim);
		Cunder_Tensor *tensor = (Cunder_Tensor *)tensor_void;

		if (tensor->tensor.getIntrusivePtr() == nullptr)
			tensor->tensor.unsafeReleaseTensorImpl();
		tensor->tensor = torch::ones(vshape, cunder::get_libtorch_dtype(dtype));
	}

	void
	cunder_tensor_zeros_allocated(void *tensor_void, int ndim, const int *shape, Cunder_DType dtype)
	{
		if (_cunder_check_initialization_param(ndim, shape, dtype) == false)
			return;

		std::vector<int64_t> vshape(shape, shape + ndim);
		Cunder_Tensor *tensor = (Cunder_Tensor *)tensor_void;

		if (tensor->tensor.getIntrusivePtr() == nullptr)
			tensor->tensor.unsafeReleaseTensorImpl();
		tensor->tensor = torch::zeros(vshape, cunder::get_libtorch_dtype(dtype));
	}

	void
	cunder_tensor_eye_allocated(void *tensor_void, int n, Cunder_DType dtype)
	{
		if (cunder::is_valid_dtype(dtype) == false)
			return;

		Cunder_Tensor *tensor = (Cunder_Tensor *)tensor_void;
		if (tensor->tensor.getIntrusivePtr() == nullptr)
			tensor->tensor.unsafeReleaseTensorImpl();
		tensor->tensor = torch::eye(n, cunder::get_libtorch_dtype(dtype));
	}

	void
	cunder_tensor_range_allocated(void *tensor_void, int start, int end, int step, Cunder_DType dtype)
	{
		if (cunder::is_valid_dtype(dtype) == false)
			return;

		Cunder_Tensor *tensor = (Cunder_Tensor *)tensor_void;
		if (tensor->tensor.getIntrusivePtr() == nullptr)
			tensor->tensor.unsafeReleaseTensorImpl();
		tensor->tensor = torch::range(start, end, step, cunder::get_libtorch_dtype(dtype));
	}

	Cunder_Tensor *
	cunder_tensor_from_data_wrap(int ndim, const int *shape, void *data, Cunder_DType dtype)
	{
		Cunder_Tensor *tensor;
		if (_cunder_check_initialization_param(ndim, shape, dtype))
			tensor = new Cunder_Tensor{};
		else
			return nullptr;

		std::vector<int64_t> vshape(shape, shape + ndim);
		tensor->tensor = torch::from_blob(data, vshape, torch::TensorOptions(cunder::get_libtorch_dtype(dtype)));
		return tensor;
	}

	Cunder_Tensor *
	cunder_tensor_from_data(int ndim, const int *shape, void *data, Cunder_DType dtype, void (*free)(void *))
	{
		Cunder_Tensor *tensor;
		if (_cunder_check_initialization_param(ndim, shape, dtype))
			tensor = new Cunder_Tensor{};
		else
			return nullptr;

		std::vector<int64_t> vshape(shape, shape + ndim);
		tensor->tensor = torch::from_blob(data, vshape, free, torch::TensorOptions(cunder::get_libtorch_dtype(dtype)));
		return tensor;
	}

	void
	cunder_tensor_from_data_wrap_allocated(void *tensor_void, int ndim, const int *shape, void *data, Cunder_DType dtype)
	{
		if (_cunder_check_initialization_param(ndim, shape, dtype) == false)
			return;

		std::vector<int64_t> vshape(shape, shape + ndim);
		Cunder_Tensor *tensor = (Cunder_Tensor *)tensor_void;

		if (tensor->tensor.getIntrusivePtr() == nullptr)
			tensor->tensor.unsafeReleaseTensorImpl();
		tensor->tensor = torch::from_blob(data, vshape, torch::TensorOptions(cunder::get_libtorch_dtype(dtype)));
	}

	void
	cunder_tensor_from_data_allocated(void *tensor_void, int ndim, const int *shape, void *data, Cunder_DType dtype, void (*free)(void *))
	{
		if (_cunder_check_initialization_param(ndim, shape, dtype) == false)
			return;

		std::vector<int64_t> vshape(shape, shape + ndim);
		Cunder_Tensor *tensor = (Cunder_Tensor *)tensor_void;

		if (tensor->tensor.getIntrusivePtr() == nullptr)
			tensor->tensor.unsafeReleaseTensorImpl();
		tensor->tensor = torch::from_blob(data, vshape, free, torch::TensorOptions(cunder::get_libtorch_dtype(dtype)));
	}

	void
	cunder_tensor_to(Cunder_Tensor *tensor, Cunder_DType dtype)
	{
		if (tensor == nullptr)
			return;

		tensor->tensor = tensor->tensor.toType(cunder::get_libtorch_dtype(dtype));
	}

	void
	cunder_tensor_to_allocated(void *tensor_void, Cunder_DType dtype)
	{
		Cunder_Tensor *tensor = (Cunder_Tensor *)tensor_void;
		if (tensor == nullptr)
			return;

		if (tensor->tensor.getIntrusivePtr() == nullptr)
			tensor->tensor.unsafeReleaseTensorImpl();
		tensor->tensor = tensor->tensor.toType(cunder::get_libtorch_dtype(dtype));
	}

	void
	cunder_tensor_print(const Cunder_Tensor *tensor)
	{
		if (tensor == nullptr)
			return;
		print(tensor->tensor);
		printf("\n");
	}

	void
	cunder_tensor_print_allocated(const void *tensor_void)
	{
		auto tensor = (Cunder_Tensor *)tensor_void;
		cunder_tensor_print(tensor);
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
	cunder_tensor_shape(const Cunder_Tensor *tensor, int64_t *out_shape)
	{
		int d = 0;
		for (int64_t size : tensor->tensor.sizes())
			out_shape[d++] = size;
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

	Cunder_DType
	cunder_tensor_type_allocated(const void *tensor_void)
	{
		auto tensor = (Cunder_Tensor *)tensor_void;
		return cunder_tensor_type(tensor);
	}

	int64_t
	cunder_tensor_ndim_allocated(const void *tensor_void)
	{
		auto tensor = (Cunder_Tensor *)tensor_void;
		return cunder_tensor_ndim(tensor);
	}

	void
	cunder_tensor_shape_allocated(const void *tensor_void, int64_t *out_shape)
	{
		auto tensor = (Cunder_Tensor *)tensor_void;
		cunder_tensor_shape(tensor, out_shape);
	}

	int64_t
	cunder_tensor_numel_allocated(const void *tensor_void)
	{
		auto tensor = (Cunder_Tensor *)tensor_void;
		return cunder_tensor_numel(tensor);
	}

	int64_t
	cunder_tensor_dim_size_allocated(const void *tensor_void, int64_t dim)
	{
		auto tensor = (Cunder_Tensor *)tensor_void;
		return cunder_tensor_dim_size(tensor, dim);
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

	Cunder_Module *
	cunder_module_load(const char *filename)
	{
		torch::jit::Module module;

		try
		{
			module = torch::jit::load(filename);
		} catch (const c10::Error &e)
		{
			printf("%s\n", e.msg().c_str());
			printf("%s\n", e.what());
			return nullptr;
		}

		Cunder_Module *cunder_module = new Cunder_Module();
		cunder_module->module = module;
		return cunder_module;
	}

	void
	cunder_module_dump(const Cunder_Module *module, bool print_method_bodies, bool print_attr_values, bool print_param_values)
	{
		if (module == nullptr)
			return;

		module->module.dump(print_method_bodies, print_attr_values, print_param_values);
	}

	void
	cunder_module_load_allocated(const char *filename, void *module_void)
	{
		torch::jit::Module module;

		try
		{
			module = torch::jit::load(filename);
		} catch (const c10::Error &e)
		{
			printf("%s\n", e.msg().c_str());
			printf("%s\n", e.what());
			return;
		}

		Cunder_Module *cunder_module = (Cunder_Module *)module_void;
		cunder_module->module = module;
	}

	void
	cunder_module_dump_allocated(const void *module_void, bool print_method_bodies, bool print_attr_values, bool print_param_values)
	{
		Cunder_Module *module = (Cunder_Module *)module_void;
		cunder_module_dump(module);
	}

	void
	cunder_tensor_print_attrs(Cunder_Tensor *tensor)
	{
		int64_t ndim = tensor->tensor.dim();
		std::cout << "dim " << ndim << std::endl;
		std::cout << "sizes " << tensor->tensor.sizes() << std::endl;
		std::cout << "dtype " << tensor->tensor.dtype() << std::endl;
		std::cout << "cunder_type " << cunder_tensor_type(tensor) << std::endl;
		int64_t *out_shape = new int64_t[ndim];
		cunder_tensor_shape(tensor, out_shape);
		for (int64_t d = 0; d < ndim; ++d)
			std::cout << out_shape[d] << ' ';
		delete[] out_shape;
		std::cout << std::endl;
	}

	void
	cunder_tensor_print_attrs_allocated(void *tensor_void)
	{
		Cunder_Tensor *tensor = (Cunder_Tensor *)tensor_void;
		if (tensor->tensor.getIntrusivePtr() == nullptr)
			tensor->tensor.unsafeReleaseTensorImpl();
		cunder_tensor_print_attrs(tensor);
	}

} // extern "C"