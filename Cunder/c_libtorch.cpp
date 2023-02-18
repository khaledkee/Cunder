#include "c_libtorch.h"

namespace dtype
{
    bool is_valid_dtype(Cunder_DType dtype)
    {
        if ((dtype == Cunder_kUint8) || (dtype == Cunder_kInt8) || (dtype == Cunder_kInt16) || (dtype == Cunder_kInt32) || (dtype == Cunder_kInt64) || (dtype == Cunder_kFloat16) || (dtype == Cunder_kFloat32) || (dtype == Cunder_kFloat64))
        {
            return true;
        }

        return false;
    }

    // Get torch native dtype.
    constexpr auto get_libtorch_dtype(Cunder_DType dtype)
    {
        switch (dtype)
        {
        case Cunder_kInvalid:
            throw std::invalid_argument("Unknown dtype");

        case Cunder_kUint8:
            return torch::kUInt8;

        case Cunder_kInt8:
            return torch::kInt8;

        case Cunder_kInt16:
            return torch::kInt16;

        case Cunder_kInt32:
            return torch::kInt32;

        case Cunder_kInt64:
            return torch::kInt64;

        case Cunder_kFloat16:
            return torch::kFloat16;

        case Cunder_kFloat32:
            return torch::kFloat32;

        case Cunder_kFloat64:
            return torch::kFloat64;
        }
    }
} // namespace dtype

extern "C"
{

    void cunder_torch_version(int *major, int *minor, int *patch)
    {
        if (major)
        {
            (*major) = TORCH_VERSION_MAJOR;
        }
        if (minor)
        {
            (*minor) = TORCH_VERSION_MINOR;
        }
        if (patch)
        {
            (*patch) = TORCH_VERSION_PATCH;
        }
    }

    int cunder_cuda_is_available()
    {
        return torch::cuda::is_available();
    }

    int delete_cunder_at_Tensor(cunder_at_Tensor *obj)
    {
        if (obj == NULL)
        {
            return -1;
        }

        delete obj->data;

        free(obj);

        return 0; // success
    }

    // torch::ones
    cunder_at_Tensor *cunder_torch_ones(int ndim, int *shape, Cunder_DType dtype)
    {
        if (shape == NULL)
        {
            return NULL;
        }

        if (!dtype::is_valid_dtype(dtype))
        {
            return NULL;
        }

        cunder_at_Tensor *tensor = reinterpret_cast<cunder_at_Tensor *>(malloc(sizeof(cunder_at_Tensor)));

        tensor->dtype = dtype;
        tensor->ndim = ndim;

        std::vector<int64_t> vshape;
        for (size_t i = 0; i < ndim; i++)
        {
            tensor->shape[i] = shape[i];
            vshape.push_back(shape[i]);
        }

        TensorData *data = new TensorData();
        data->tensor = torch::ones(vshape, dtype::get_libtorch_dtype(dtype));

        tensor->data = data;

        return tensor;
    }

    cunder_at_Tensor *cunder_torch_ones_1d(int sz, Cunder_DType dtype)
    {

        int shape[1];
        shape[0] = sz;

        return cunder_torch_ones(1, shape, dtype);
    }

    cunder_at_Tensor *cunder_torch_ones_2d(int sz0, int sz1, Cunder_DType dtype)
    {

        int shape[2];
        shape[0] = sz0;
        shape[1] = sz1;

        return cunder_torch_ones(2, shape, dtype);
    }

    cunder_at_Tensor *cunder_torch_ones_3d(int sz0, int sz1, int sz2, Cunder_DType dtype)
    {

        int shape[3];
        shape[0] = sz0;
        shape[1] = sz1;
        shape[2] = sz2;

        return cunder_torch_ones(3, shape, dtype);
    }

    cunder_at_Tensor *cunder_torch_ones_4d(int sz0, int sz1, int sz2, int sz3, Cunder_DType dtype)
    {

        int shape[4];
        shape[0] = sz0;
        shape[1] = sz1;
        shape[2] = sz2;
        shape[3] = sz3;

        return cunder_torch_ones(4, shape, dtype);
    }

    // torch::zeros
    cunder_at_Tensor *cunder_torch_zeros(int ndim, int *shape, Cunder_DType dtype)
    {
        if (shape == NULL)
        {
            return NULL;
        }

        if (!dtype::is_valid_dtype(dtype))
        {
            return NULL;
        }

        cunder_at_Tensor *tensor = reinterpret_cast<cunder_at_Tensor *>(malloc(sizeof(cunder_at_Tensor)));

        tensor->dtype = dtype;
        tensor->ndim = ndim;

        std::vector<int64_t> vshape;
        for (size_t i = 0; i < ndim; i++)
        {
            tensor->shape[i] = shape[i];
            vshape.push_back(shape[i]);
        }

        TensorData *data = new TensorData();
        data->tensor = torch::zeros(vshape, dtype::get_libtorch_dtype(dtype));

        tensor->data = data;

        return tensor;
    }

    cunder_at_Tensor *cunder_torch_zeros_1d(int sz, Cunder_DType dtype)
    {

        int shape[1];
        shape[0] = sz;

        return cunder_torch_zeros(1, shape, dtype);
    }

    cunder_at_Tensor *cunder_torch_zeros_2d(int sz0, int sz1, Cunder_DType dtype)
    {

        int shape[2];
        shape[0] = sz0;
        shape[1] = sz1;

        return cunder_torch_zeros(2, shape, dtype);
    }

    cunder_at_Tensor *cunder_torch_zeros_3d(int sz0, int sz1, int sz2, Cunder_DType dtype)
    {

        int shape[3];
        shape[0] = sz0;
        shape[1] = sz1;
        shape[2] = sz2;

        return cunder_torch_zeros(3, shape, dtype);
    }

    cunder_at_Tensor *cunder_torch_zeros_4d(int sz0, int sz1, int sz2, int sz3, Cunder_DType dtype)
    {

        int shape[4];
        shape[0] = sz0;
        shape[1] = sz1;
        shape[2] = sz2;
        shape[3] = sz3;

        return cunder_torch_zeros(4, shape, dtype);
    }

    // torch::eye()
    cunder_at_Tensor *cunder_torch_eye(int n, Cunder_DType dtype)
    {
        if (!dtype::is_valid_dtype(dtype))
        {
            return NULL;
        }

        cunder_at_Tensor *tensor = reinterpret_cast<cunder_at_Tensor *>(malloc(sizeof(cunder_at_Tensor)));

        tensor->dtype = dtype;

        TensorData *data = new TensorData();
        data->tensor = torch::eye(n, dtype::get_libtorch_dtype(dtype));

        tensor->data = data;
        tensor->ndim = n;
        return tensor;
    }

    // torch::range
    cunder_at_Tensor *cunder_torch_range(int start, int end, int step, Cunder_DType dtype)
    {
        if (!dtype::is_valid_dtype(dtype))
        {
            return NULL;
        }

        cunder_at_Tensor *tensor = reinterpret_cast<cunder_at_Tensor *>(malloc(sizeof(cunder_at_Tensor)));
        tensor->dtype = dtype;
        tensor->ndim = 1;

        TensorData *data = new TensorData();
        data->tensor = torch::range(start, end, step, dtype::get_libtorch_dtype(dtype));

        tensor->data = data;

        return tensor;
    }

} // extern "C"
