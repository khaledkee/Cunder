#include "c_libtorch.h"

//#include <torch/all.h>
//#include <torch/script.h>

namespace
{
	Torch_Version
	c_torch_version()
	{
		return Torch_Version{};
		// TODO: defined in torch/all.h
//		return Torch_Version{TORCH_VERSION_MAJOR, TORCH_VERSION_MINOR, TORCH_VERSION_PATCH};
	}
} // namespace

extern "C"
{

} // extern "C"