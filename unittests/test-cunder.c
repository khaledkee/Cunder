#include "acutest.h"
#include "c_libtorch.h"

void test_tutorial()
{
  TEST_CHECK_(1 == 1, "bora");
}

void test_version()
{
  int major, minor, patch;

  cunder_torch_version(&major, &minor, &patch);

  printf("version: %d.%d.%d\n", major, minor, patch);
}

void test_ones_f64_1()
{
  cunder_at_Tensor *tp = cunder_torch_ones_1d(9, Cunder_kFloat64);
  delete_cunder_at_Tensor(tp);
}

void test_ones_f64_2()
{
  cunder_at_Tensor *tp = cunder_torch_ones_2d(9, 8, Cunder_kFloat32);
  delete_cunder_at_Tensor(tp);
}

void test_ones_f64_3()
{
  cunder_at_Tensor *tp = cunder_torch_ones_3d(9, 8, 7, Cunder_kInt32);
  delete_cunder_at_Tensor(tp);
}

void test_ones_f64_4()
{
  cunder_at_Tensor *tp = cunder_torch_ones_4d(9, 8, 8, 8, Cunder_kInt16);
  delete_cunder_at_Tensor(tp);
}

void test_zeros_f64_1()
{
  cunder_at_Tensor *tp = cunder_torch_zeros_1d(9, Cunder_kFloat64);
  delete_cunder_at_Tensor(tp);
}

void test_zeros_f64_2()
{
  cunder_at_Tensor *tp = cunder_torch_zeros_2d(9, 8, Cunder_kFloat32);
  delete_cunder_at_Tensor(tp);
}

void test_zeros_f64_3()
{
  cunder_at_Tensor *tp = cunder_torch_zeros_3d(9, 8, 7, Cunder_kInt32);
  delete_cunder_at_Tensor(tp);
}

void test_zeros_f64_4()
{
  cunder_at_Tensor *tp = cunder_torch_zeros_4d(9, 8, 8, 8, Cunder_kInt16);
  delete_cunder_at_Tensor(tp);
}

void test_torch_eye()
{
  cunder_at_Tensor *tp = cunder_torch_eye(8, Cunder_kInt8);
  delete_cunder_at_Tensor(tp);
}

void test_cuda()
{
  int has_cuda = cunder_cuda_is_available();
  printf("cuda? %d\n", has_cuda);
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
    {"eye", test_torch_eye},
    {"version", test_version},
    {"cuda", test_cuda},
    {NULL, NULL}};