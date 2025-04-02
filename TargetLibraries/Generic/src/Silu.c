#include "DeeployBasicMath.h"
#include <math.h>

void Silu(float *data_in, float *data_out, uint32_t size) {
  for (uint32_t i = 0; i < size; i++) {
    data_out[i] = data_in[i] / (1 + expf(-data_in[i]));
  }
}
