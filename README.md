# Cunder :zap: :zap:

Experimental C binding for LibTorch

# Quick start

# Usage

see `playground/main.cpp`

```cpp
#include <cunder/c_libtorch.h>
#include <iostream>

int main()
{
	// create tensor
	float tensor_data[] = {1, 9, 1};
	int tensor_data_shape[] = {3};
	auto cunder_data_tensor = cunder_tensor_from_data_wrap(1, tensor_data_shape, tensor_data, Cunder_DType::Cunder_Float32);
	
	// print the tensor
	printf("Cunder Tensor external data: \n");
	cunder_tensor_print(cunder_data_tensor);
	cunder_tensor_print_attributes(cunder_data_tensor);

	// access tensor data
	auto *tensor_accessor = cunder_tensor_accessor_f32(cunder_data_tensor);
	for (size_t i = 0; i < 3; i++)
		std::cout << tensor_accessor[i] << ',';
	std::cout << std::endl;

	// free the tensor
	cunder_tensor_free(cunder_data_tensor);
	return 0;
}
```

# Structure

NOTE: add LibTorch dlls and include files in external folder OR use CMake find(torch) and add argument `DCMAKE_PREFIX_PATH`

# Roadmap

- [x] `torch::version()`
- [x] Create tensor from data
  - [x] given shape create zero tensor
  - [x] given some data (pointer) with shape create tensor
    - [x] wrapping the data
    - [x] cloning the data
  - Supported types:
    - [x] bool
    - [x] uint8
    - [x] int8
    - [x] int32 (int)
    - [x] int64 (long)
    - [x] float32
    - [x] float64
  - [x] access tensor data
- [ ] Torch script jit model
  - [x] Module struct `torch::jit::Module`
  - [x] load Module `torch::jit::load()`
  - [ ] call `eval()` on Module
  - [ ] run Module on cpu (call `forward()` with tensors)
- [ ] Add support to external libraries:
  - [ ] torch_sparse
  - [ ] torch_scatter
