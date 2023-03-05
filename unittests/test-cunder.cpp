#include "acutest.h"
#include "c_libtorch.h"

void
test_tutorial()
{
	TEST_CHECK_(1 == 1, "bora");
}

void
test_version()
{
	Torch_Version version = cunder_torch_version();

	printf("version: %d.%d.%d\n", version.major, version.minor, version.patch);
}

void
test_ones_f64_1()
{
	int shape[] = {9};
	Cunder_Tensor *tp = cunder_tensor_ones(1, shape, Cunder_Float64);
	cunder_tensor_free(tp);
}

void
test_ones_f64_2()
{
	int shape[] = {9, 8};
	Cunder_Tensor *tp = cunder_tensor_ones(2, shape, Cunder_Float32);
	cunder_tensor_free(tp);
}

void
test_ones_f64_3()
{
	int shape[] = {9, 8, 7};
	Cunder_Tensor *tp = cunder_tensor_ones(3, shape, Cunder_Int32);
	cunder_tensor_free(tp);
}

void
test_ones_f64_4()
{
	int shape[] = {9, 8, 8, 8};
	Cunder_Tensor *tp = cunder_tensor_ones(4, shape, Cunder_Int16);
	cunder_tensor_free(tp);
}

void
test_zeros_f64_1()
{
	int shape[] = {9};
	Cunder_Tensor *tp = cunder_tensor_zeros(1, shape, Cunder_Float64);
	cunder_tensor_free(tp);
}

void
test_zeros_f64_2()
{
	int shape[] = {9, 8};
	Cunder_Tensor *tp = cunder_tensor_zeros(2, shape, Cunder_Float32);
	cunder_tensor_free(tp);
}

void
test_zeros_f64_3()
{
	int shape[] = {9, 8, 7};
	Cunder_Tensor *tp = cunder_tensor_zeros(3, shape, Cunder_Int32);
	cunder_tensor_free(tp);
}

void
test_zeros_f64_4()
{
	int shape[] = {9, 8, 8, 8};
	Cunder_Tensor *tp = cunder_tensor_zeros(3, shape, Cunder_Int16);
	cunder_tensor_free(tp);
}

void
test_tensor_eye()
{
	Cunder_Tensor *tp = cunder_tensor_eye(8, Cunder_Int8);
	cunder_tensor_free(tp);
}

TEST_LIST = {
	{"test", test_tutorial},
	{"ones_1d", test_ones_f64_1},
	{"ones_2d", test_ones_f64_2},
	{"ones_3d", test_ones_f64_3},
	{"ones_4d", test_ones_f64_4},
	{"zeros_1d", test_zeros_f64_1},
	{"zeros_2d", test_zeros_f64_2},
	{"zeros_3d", test_zeros_f64_3},
	{"zeros_4d", test_zeros_f64_4},
	{"eye", test_tensor_eye},
	{"version", test_version},
	{NULL, NULL}};