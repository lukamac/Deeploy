#ifndef __DEEPLOY_BASIC_MATH_DEQUANTSHIFT_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_DEQUANTSHIFT_KERNEL_HEADER_

#include <stdint.h>


void DequantShift_s8(int8_t *data_in, int32_t size, int32_t mul,
                     int32_t add, float *data_out, int32_t log2D);
void DequantShift_u8(uint8_t *data_in, int32_t size, int32_t mul,
                     int32_t add, float *data_out, int32_t log2D);

#endif // __DEEPLOY_BASIC_MATH_DEQUANTSHIFT_KERNEL_HEADER_
