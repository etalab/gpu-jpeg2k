#include "my_common.h"

void* my_malloc (size_t size) {
	return calloc(size, 1);
}