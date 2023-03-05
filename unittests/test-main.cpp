#define DOCTEST_CONFIG_IMPLEMENT
#define LOG_LEVEL 1 // FATAL logging

#include <doctest/doctest.h>
#include "c_libtorch.h"

void *
my_alloc(size_t size, uint8_t alignment)
{
	if (size == 0)
		return nullptr;
	void *data = _aligned_malloc(size, alignment);
	if (LOG_LEVEL >= 4) // INFO logging
		printf("allocating %lld %u at %p\n", size, alignment, data);
	memset(data, 0, size);
	return data;
}

void
my_free(void *data)
{
	if (LOG_LEVEL >= 4) // INFO logging
		printf("freeing %p\n", data);
	_aligned_free(data);
}

int
main(int argc, char **argv)
{
	Torch_Version version = cunder_torch_version();
	printf("Torch Version: %d.%d.%d\n\n", version.major, version.minor, version.patch);

	// setUp cpu allocator
	auto allocator = cunder_set_cpu_allocator(my_alloc, my_free);

	doctest::Context context;
	int res = context.run();

	// tearDown cpu allocator
	cunder_allocator_free(allocator);

	return res;
}