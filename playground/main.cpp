#include "c_libtorch.h"

#include <torch/torch.h>
#include <iostream>

template<typename T>
void
pretty_print(const std::string &info, T &&data)
{
	std::cout << info << std::endl;
	std::cout << data << std::endl << std::endl;
}

int
main(int argc, char *argv[])
{
	Torch_Version version = cunder_torch_version();
	printf("Cunder torch version: %d.%d.%d\n", version.major, version.minor, version.patch);

	// Create an eye tensor
	auto eye_tensor = torch::eye(3);
	auto *cunder_eye_tensor = cunder_tensor_eye(3, Cunder_DType::Cunder_Float64);
	pretty_print("libtorch Eye tensor: ", eye_tensor);
	printf("Cunder Eye tensor: \n");
	cunder_tensor_print(cunder_eye_tensor);
	cunder_tensor_free(cunder_eye_tensor);

	// Create range tensor
	auto range_tensor = torch::range(1, 9, 1);
	auto cunder_range_tensor = cunder_tensor_range(1, 9, 1, Cunder_DType::Cunder_Float64);
	pretty_print("Tensor range 1x9: ", range_tensor);
	printf("Cunder Range tensor: \n");
	cunder_tensor_print(cunder_range_tensor);
	cunder_tensor_free(cunder_range_tensor);

	// Create zeros tensor
	int zeros_shape[] = {1, 4};
	auto zeros_tensor = torch::zeros({1, 4});
	auto cunder_zeros_tensor = cunder_tensor_zeros(2, zeros_shape, Cunder_DType::Cunder_Float32);
	pretty_print("Tensor zeros : ", zeros_tensor);
	printf("Cunder Tensor zeros: \n");
	cunder_tensor_print(cunder_zeros_tensor);
	cunder_tensor_free(cunder_zeros_tensor);

	// Create ones tensor
	int ones_shape[] = {1, 9, 1};
	auto ones_tensor = torch::ones({1, 9, 1});
	auto cunder_ones_tensor = cunder_tensor_ones(3, ones_shape, Cunder_DType::Cunder_Float32);
	pretty_print("Tensor ones: ", ones_tensor);
	printf("Cunder Tensor ones: \n");
	cunder_tensor_print(cunder_ones_tensor);
	cunder_tensor_free(cunder_ones_tensor);

	// Create tensor from data
	float tensor_data[] = {1, 9, 1};
	int tensor_data_shape[] = {3};
	auto data_tensor = torch::from_blob(tensor_data, {3});
	auto cunder_data_tensor = cunder_tensor_from_data_wrap(1, tensor_data_shape, tensor_data, Cunder_DType::Cunder_Float32);
	pretty_print("Tensor external data: ", data_tensor);
	printf("Cunder Tensor external data: \n");
	cunder_tensor_print(cunder_data_tensor);

	// access tensor data
	auto *tensor_accessor = cunder_tensor_accessor_f32(cunder_data_tensor);
	for (size_t i = 0; i < 3; i++)
		std::cout << tensor_accessor[i] << ",\n"[i == 2];

	cunder_tensor_free(cunder_data_tensor);

	// cunder_module
	Cunder_Module *cunder_module = cunder_module_load("D:\\model.pt");
	cunder_module_dump(cunder_module, false, false, false);
	cunder_module_free(cunder_module);

	int64_t size, alignment;
	cunder_get_tensor_size_alignment(size, alignment);

	Cunder_Tensor *allocated_cunder_zeros_tensor = (Cunder_Tensor*)_aligned_malloc(size, alignment);
	memset(allocated_cunder_zeros_tensor, 0, size);
	cunder_tensor_zeros_allocated(allocated_cunder_zeros_tensor, 2, zeros_shape, Cunder_DType::Cunder_Float32);
	cunder_tensor_print_attributes(allocated_cunder_zeros_tensor);
	cunder_tensor_print(allocated_cunder_zeros_tensor);
	_aligned_free(allocated_cunder_zeros_tensor);

	Cunder_Tensor *allocated_cunder_ones_tensor = (Cunder_Tensor*)_aligned_malloc(size, alignment);
	memset(allocated_cunder_ones_tensor, 0, size);
	cunder_tensor_ones_allocated(allocated_cunder_ones_tensor, 3, ones_shape, Cunder_DType::Cunder_Float32);
	cunder_tensor_print_attributes(allocated_cunder_ones_tensor);
	cunder_tensor_print(allocated_cunder_ones_tensor);
	_aligned_free(allocated_cunder_ones_tensor);

	Cunder_Tensor *allocated_cunder_eye_tensor = (Cunder_Tensor*)_aligned_malloc(size, alignment);
	memset(allocated_cunder_eye_tensor, 0, size);
	cunder_tensor_eye_allocated(allocated_cunder_eye_tensor, 3, Cunder_DType::Cunder_Float32);
	cunder_tensor_print_attributes(allocated_cunder_eye_tensor);
	cunder_tensor_print(allocated_cunder_eye_tensor);
	_aligned_free(allocated_cunder_eye_tensor);

	Cunder_Tensor *allocated_cunder_range_tensor = (Cunder_Tensor*)_aligned_malloc(size, alignment);
	memset(allocated_cunder_range_tensor, 0, size);
	cunder_tensor_range_allocated(allocated_cunder_range_tensor, 1, 9, 1, Cunder_DType::Cunder_Float32);
	cunder_tensor_print_attributes(allocated_cunder_range_tensor);
	cunder_tensor_print(allocated_cunder_range_tensor);
	_aligned_free(allocated_cunder_range_tensor);

	Cunder_Tensor *allocated_cunder_data_tensor = (Cunder_Tensor*)_aligned_malloc(size, alignment);
	memset(allocated_cunder_data_tensor, 0, size);
	cunder_tensor_from_data_allocated(
		allocated_cunder_data_tensor, 1, tensor_data_shape, tensor_data, Cunder_DType::Cunder_Float32, nullptr);
	cunder_tensor_print_attributes(allocated_cunder_data_tensor);
	cunder_tensor_print(allocated_cunder_data_tensor);

	Cunder_Tensor *allocated_cunder_data_tensor_clone = (Cunder_Tensor*)_aligned_malloc(size, alignment);
	memset(allocated_cunder_data_tensor_clone, 0, size);
	cunder_tensor_clone_allocated(allocated_cunder_data_tensor_clone, allocated_cunder_data_tensor);
	cunder_tensor_print_attributes(allocated_cunder_data_tensor_clone);
	cunder_tensor_print(allocated_cunder_data_tensor_clone);
	_aligned_free(allocated_cunder_data_tensor_clone);
	_aligned_free(allocated_cunder_data_tensor);

	return 0;
}
