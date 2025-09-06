#include "DeeployBasicMath.h"
#include <math.h>

static inline float silu_naive(float x) {
  return x / (1.0f + expf(-x));
}

static inline float silu_stable(float x) {
  if (x >= 0.0f) {
    return x / (1.0f + expf(-x));
  } else {
    const float z = expf(x);
    return x * z / (1.0f + z);
  }
}

void Silu(float *data_in, float *data_out, uint32_t size) {
  for (uint32_t i = 0; i < size; i++) {
    data_out[i] = silu_stable(data_in[i]);
  }
}
