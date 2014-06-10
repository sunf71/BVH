#include "cuda_runtime.h"


__declspec(dllexport) cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
