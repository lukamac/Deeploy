#include "DeeployBasicMath.h"


void DequantShift_s8(int8_t *data_in, int32_t size, int32_t mul,
                     int32_t add, float *data_out, int32_t log2D) {
  for (int i = 0; i < size; i++) {
    data_out[i] = (float)(((int32_t)data_in[i] << log2D) - add) / (float)mul;
  }
}

void DequantShift_u8(uint8_t *data_in, int32_t size, int32_t mul,
                     int32_t add, float *data_out, int32_t log2D) {
  for (int i = 0; i < size; i++) {
    data_out[i] = (float)(((int32_t)data_in[i] << log2D) - add) / (float)mul;
  }
}
