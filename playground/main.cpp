#include "c_libtorch.h"

#include <torch/torch.h>
#include <iostream>

template <typename T>
void pretty_print(const std::string &info, T &&data)
{
    std::cout << info << std::endl;
    std::cout << data << std::endl
              << std::endl;
}

int main()
{
    int major, minor, patch;
    cunder_torch_version(&major, &minor, &patch);
    printf("Cunder torch version: %d.%d.%d\n", major, minor, patch);
    printf("Cunder CUDA: %d\n", cunder_cuda_is_available());

    // Create an eye tensor
    torch::Tensor eye_tensor = torch::eye(3);
    auto *eye_cunder_tensor = cunder_torch_eye(3, Cunder_DType::Cunder_kFloat64);
    pretty_print("libtorch Eye tensor: ", eye_tensor);
    pretty_print("Cunder Eye tensor: ", eye_cunder_tensor->data->tensor);

    // Create range tensor 
    auto range_tensor = torch::range(1, 9, 1);
    auto cunder_range_tensor = cunder_torch_range(1, 9, 1, Cunder_DType::Cunder_kFloat64);
    pretty_print("Tensor range 1x9: ", range_tensor);
    pretty_print("Cunder Tensor range 1x9: ", cunder_range_tensor->data->tensor);

    // Create zeros tensor
    auto zeros_tensor = torch::zeros({1, 4});
    auto cunder_zeros_tensor = cunder_torch_zeros_2d(1, 4, Cunder_DType::Cunder_kFloat16);
    pretty_print("Tensor zeros : ", zeros_tensor);
    pretty_print("Cunder Tensor zeros: ", cunder_zeros_tensor->data->tensor);

    // Create ones tensor
    auto ones_tensor = torch::ones({1, 9, 1});
    auto cunder_ones_tensor = cunder_torch_ones_3d(1, 9, 1, Cunder_DType::Cunder_kFloat16);
    pretty_print("Tensor ones: ", ones_tensor);
    pretty_print("Cunder Tensor ones: ", cunder_ones_tensor->data->tensor);

    return 0;
}