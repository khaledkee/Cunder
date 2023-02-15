# Cunder :zap: :zap:

Experimental C binding for LibTorch

# Quick start

# Usage

see `playground/main.cpp`

```cpp
#include <cunder/c_libtorch.h>
#include <cunder/c_pytorch_sparse.h>
#include <cunder/c_pytorch_scatter.h>

int main()
{
	// TODO: write usage here
	return 0;
}

```

# Structure

NOTE: add LibTorch dlls and include files in external folder OR use CMake find(torch) and add argument `DCMAKE_PREFIX_PATH`

# Roadmap

- [ ] `torch::version()`
- [ ] Create tensor from data
  - [ ] given shape create zero tensor
  - [ ] given some data (pointer) with shape create tensor
    - [ ] wrapping the data
    - [ ] cloning the data
      - [ ] Memory allocation for new tensor data
  - Supported types:
    - [ ] float32
    - [ ] bool
    - [ ] int8
    - [ ] uint8
    - [ ] int32 (int)
    - [ ] int64 (long)
    - [ ] [MAYBE] bfloat16
  - [ ] indexing tensor (get and set)
  - [ ] call `item()` on tensor to get raw data
- [ ] Torch script jit model
  - [ ] Module struct `torch::jit::Module`
  - [ ] load Module `torch::jit::load()`
  - [ ] call `eval()` on Module
  - [ ] run Module on cpu (call `forward()` with tensors)
- [ ] Add support to external libraries:
  - [ ] torch_sparse
  - [ ] torch_scatter
