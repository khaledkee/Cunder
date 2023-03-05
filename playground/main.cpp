#include "c_libtorch.h"

#include <iostream>

template<typename T>
void
pretty_print(const std::string &info, T &&data)
{
	std::cout << info << std::endl;
	std::cout << data << std::endl << std::endl;
}

void *
my_alloc(size_t size, uint8_t alignment)
{
	if (size == 0)
		return nullptr;
	void *data = _aligned_malloc(size, alignment);
	printf("allocating %lld %u at %p\n", size, alignment, data);
	memset(data, 0, size);
	return data;
}

void
my_free(void *data)
{
	printf("freeing %p\n", data);
	_aligned_free(data);
}

int
main(int argc, char *argv[])
{
	Torch_Version version = cunder_torch_version();
	printf("Cunder torch version: %d.%d.%d\n", version.major, version.minor, version.patch);

	Cunder_Allocator *allocator = cunder_set_cpu_allocator(my_alloc, my_free);

	// Create an eye tensor
	auto *cunder_eye_tensor = cunder_tensor_eye(3, Cunder_DType::Cunder_Float64);
	printf("Cunder Eye tensor: \n");
	cunder_tensor_print(cunder_eye_tensor);
	cunder_tensor_free(cunder_eye_tensor);

	// Create range tensor
	auto cunder_range_tensor = cunder_tensor_range(1, 9, 1, Cunder_DType::Cunder_Float64);
	printf("Cunder Range tensor: \n");
	cunder_tensor_print(cunder_range_tensor);
	cunder_tensor_free(cunder_range_tensor);

	// Create zeros tensor
	int zeros_shape[] = {1, 4};
	auto cunder_zeros_tensor = cunder_tensor_zeros(2, zeros_shape, Cunder_DType::Cunder_Float32);
	printf("Cunder Tensor zeros: \n");
	cunder_tensor_print(cunder_zeros_tensor);
	cunder_tensor_free(cunder_zeros_tensor);

	// Create ones tensor
	int ones_shape[] = {1, 9, 1};
	auto cunder_ones_tensor = cunder_tensor_ones(3, ones_shape, Cunder_DType::Cunder_Float32);
	printf("Cunder Tensor ones: \n");
	cunder_tensor_print(cunder_ones_tensor);
	cunder_tensor_free(cunder_ones_tensor);

	// Create tensor from data
	float tensor_data[] = {1, 9, 1, 3, 2, 5};
	int tensor_data_shape[] = {/* batch */ 3, /* channel */ 2};
	auto data_tensor = torch::from_blob(tensor_data, {3, 2});
	auto cunder_data_tensor = cunder_tensor_from_data(2, tensor_data_shape, tensor_data, Cunder_DType::Cunder_Float32);
	pretty_print("Tensor external data: ", data_tensor);
	printf("Cunder Tensor external data: \n");
	cunder_tensor_print(cunder_data_tensor);

	// access tensor data
	auto *tensor_accessor = cunder_tensor_accessor_f32(cunder_data_tensor);
	for (size_t i = 0; i < 3; i++)
		std::cout << tensor_accessor[i] << ",\n"[i == 2];

	// clone tensor
	auto cunder_data_tensor_clone = cunder_tensor_clone(cunder_data_tensor);
	printf("Cunder Tensor external data clone: \n");
	cunder_tensor_print(cunder_data_tensor_clone);
	cunder_tensor_free(cunder_data_tensor);
	cunder_tensor_free(cunder_data_tensor_clone);

	// cunder_module
	Cunder_Module *cunder_module = cunder_module_load(CUNDER_DATA_DIR "\\model.pt");
	cunder_module_dump(cunder_module, false, false, false);

	// module forward
	cunder_module_eval(cunder_module);
	std::cout << "Module input count: " << cunder_module_input_num(cunder_module) << std::endl;

	size_t output_count;
	Cunder_Tensor *output_tensors = cunder_module_forward(cunder_module, 1, cunder_data_tensor, output_count);
	std::cout << "Module output count: " << output_count << std::endl;
	cunder_tensor_print_attributes(output_tensors);

	cunder_tensor_free(cunder_data_tensor);
	cunder_module_free(cunder_module);

	float tensor_data_2[] = {1, 9, 0, 3, 2};
	int tensor_data_shape_2[] = {/* batch */ 5, /* channel */ 1};
	auto cunder_data_tensor_2 = cunder_tensor_from_data(2, tensor_data_shape_2, tensor_data_2, Cunder_DType::Cunder_Float32);
	printf("Cunder Tensor external data: \n");
	cunder_tensor_print(cunder_data_tensor_2);

	float tensor_data_3[] = {0, 3, 2, 1};
	int tensor_data_shape_3[] = {/* batch */ 4, /* channel */ 1};
	auto cunder_data_tensor_3 = cunder_tensor_from_data(2, tensor_data_shape_3, tensor_data_3, Cunder_DType::Cunder_Float32);
	printf("Cunder Tensor external data: \n");
	cunder_tensor_print(cunder_data_tensor_3);

	// cunder_module 2 input 3 output
	Cunder_Module *cunder_module_2_3 = cunder_module_load(CUNDER_DATA_DIR "\\model_2_input_3_output.pt");
	cunder_module_dump(cunder_module_2_3, false, false, false);

	// module forward
	cunder_module_eval(cunder_module_2_3);
	std::cout << "Module input count: " << cunder_module_input_num(cunder_module_2_3) << std::endl;

	size_t output_count_2_3;
	Cunder_Tensor * model_2_3_inputs = cunder_tensor_allocate(2);
	cunder_tensor_array_set(model_2_3_inputs, 0, cunder_data_tensor_2);
	cunder_tensor_array_set(model_2_3_inputs, 1, cunder_data_tensor_3);
	Cunder_Tensor *output_tensors_2_3 = cunder_module_forward(cunder_module_2_3, 2, model_2_3_inputs, output_count_2_3);
	std::cout << "Module output count: " << output_count_2_3 << std::endl;
	cunder_tensor_print_attributes(output_tensors_2_3);

	cunder_tensor_free(model_2_3_inputs);
	cunder_tensor_free(cunder_data_tensor_2);
	cunder_tensor_free(cunder_data_tensor_3);
	cunder_module_free(cunder_module_2_3);

	cunder_allocator_free(allocator);
	return 0;
}
