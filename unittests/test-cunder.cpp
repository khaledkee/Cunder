#include <doctest/doctest.h>
#include "c_libtorch.h"

// Create zeros tensor
TEST_CASE("[Tensor] zeros")
{
	SUBCASE("1d")
	{
		int shape[] = {9};
		Cunder_Tensor *cunder_tensor = cunder_tensor_zeros(1, shape, Cunder_Float64);
		CHECK(cunder_tensor_numel(cunder_tensor) == 9); // elements count
		cunder_tensor_free(cunder_tensor);
	}

	SUBCASE("2d")
	{
		int shape[] = {9, 8};
		Cunder_Tensor *cunder_tensor = cunder_tensor_zeros(2, shape, Cunder_Float32);
		CHECK(cunder_tensor_numel(cunder_tensor) == 72); // 9*8
		cunder_tensor_free(cunder_tensor);
	}

	SUBCASE("3d")
	{
		int shape[] = {9, 8, 7};
		Cunder_Tensor *cunder_tensor = cunder_tensor_zeros(3, shape, Cunder_Int32);
		CHECK(cunder_tensor_numel(cunder_tensor) == 504); // 9*8*7
		cunder_tensor_free(cunder_tensor);
	}
}

// Create ones tensor
TEST_CASE("[Tensor] ones")
{
	SUBCASE("1d")
	{
		int shape[] = {9};
		auto cunder_tensor = cunder_tensor_ones(1, shape, Cunder_Float64);
		CHECK(cunder_tensor_numel(cunder_tensor) == 9); // elements count
		cunder_tensor_free(cunder_tensor);
	}

	SUBCASE("2d")
	{
		int shape[] = {9, 8};
		Cunder_Tensor *cunder_tensor = cunder_tensor_ones(2, shape, Cunder_Float32);
		CHECK(cunder_tensor_numel(cunder_tensor) == 72); // 9*8
		cunder_tensor_free(cunder_tensor);
	}

	SUBCASE("3d")
	{
		int shape[] = {9, 8, 7};
		Cunder_Tensor *cunder_tensor = cunder_tensor_ones(3, shape, Cunder_Int32);
		CHECK(cunder_tensor_numel(cunder_tensor) == 504); // 9*8*7
		cunder_tensor_free(cunder_tensor);
	}
}

// Create an eye tensor
TEST_CASE("[Tensor] eye")
{
	Cunder_Tensor *cunder_tensor = cunder_tensor_eye(8, Cunder_Int8);
	CHECK(cunder_tensor_numel(cunder_tensor) == 64); // 8*8
	cunder_tensor_free(cunder_tensor);
}

// Create range tensor
TEST_CASE("[Tensor] range")
{
	SUBCASE("1,...,9")
	{
		auto cunder_tensor = cunder_tensor_range(1, 9, 1, Cunder_DType::Cunder_Float64);
		CHECK(cunder_tensor_numel(cunder_tensor) == 9); // 1,2,...,9
		cunder_tensor_free(cunder_tensor);
	}

	SUBCASE("1,..,9 odd")
	{
		auto cunder_tensor = cunder_tensor_range(1, 9, 2, Cunder_DType::Cunder_Float64);
		CHECK(cunder_tensor_numel(cunder_tensor) == 5); // 1,3,5,7,9
		cunder_tensor_free(cunder_tensor);
	}
}

// Create tensor from data
TEST_CASE("[Tensor] data")
{
	float tensor_data[] = {1, 9, 1, 3, 2, 5};
	int tensor_data_shape[] = {/* batch */ 3, /* channel */ 2};
	auto cunder_tensor = cunder_tensor_from_data(2, tensor_data_shape, tensor_data, Cunder_DType::Cunder_Float32);
	CHECK(cunder_tensor_numel(cunder_tensor) == 6); // elements count
	cunder_tensor_free(cunder_tensor);
}

// clone tensor
TEST_CASE("[Tensor] clone")
{
	float tensor_data[] = {1, 9, 1, 3, 2, 5};
	int tensor_data_shape[] = {/* batch */ 3, /* channel */ 2};
	auto cunder_tensor = cunder_tensor_from_data(2, tensor_data_shape, tensor_data, Cunder_DType::Cunder_Float32);
	auto cloned_cunder_tensor = cunder_tensor_clone(cunder_tensor);
	cunder_tensor_free(cunder_tensor);
	cunder_tensor_free(cloned_cunder_tensor);
}

// cunder_module forward
TEST_CASE("[Module] forward")
{

	Cunder_Module *cunder_module = cunder_module_load(CUNDER_DATA_DIR "\\model_2_input_3_output.pt");

	// module forward
	cunder_module_eval(cunder_module);
	CHECK(cunder_module_input_num(cunder_module) == 2);

	Cunder_Array model_inputs = cunder_tensor_allocate(2);

	float tensor_data_2[] = {1, 9, 0, 3, 2};
	int tensor_data_shape_2[] = {/* batch */ 5, /* channel */ 1};
	auto cunder_data_tensor_2 = cunder_tensor_from_data(2, tensor_data_shape_2, tensor_data_2, Cunder_DType::Cunder_Float32);
	float tensor_data_3[] = {0, 3, 2, 1};
	int tensor_data_shape_3[] = {/* batch */ 4, /* channel */ 1};
	auto cunder_data_tensor_3 = cunder_tensor_from_data(2, tensor_data_shape_3, tensor_data_3, Cunder_DType::Cunder_Float32);
	cunder_tensor_array_set(model_inputs, 0, cunder_data_tensor_2);
	cunder_tensor_array_set(model_inputs, 1, cunder_data_tensor_3);

	Cunder_Array output_tensors = cunder_module_forward(cunder_module, model_inputs);
	CHECK(output_tensors.length == 3);
	CHECK(cunder_tensor_numel(cunder_tensor_array_get(output_tensors, 0)) == 15);
	CHECK(cunder_tensor_numel(cunder_tensor_array_get(output_tensors, 1)) == 12);
	CHECK(cunder_tensor_numel(cunder_tensor_array_get(output_tensors, 2)) == 30);

	cunder_array_free(model_inputs);
	cunder_tensor_free(cunder_data_tensor_2);
	cunder_tensor_free(cunder_data_tensor_3);
	cunder_array_free(output_tensors);
	cunder_module_free(cunder_module);
}