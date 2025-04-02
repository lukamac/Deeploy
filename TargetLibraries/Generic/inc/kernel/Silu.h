#ifndef __DEEPLOY_BASIC_MATH_SILU_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_SILU_KERNEL_HEADER_

#include <stdint.h>

void Silu(float *data_in, float *data_out, uint32_t size);

#endif // __DEEPLOY_BASIC_MATH_SILU_KERNEL_HEADER_
