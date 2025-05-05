/* =====================================================================
 * Title:        Convolution_float32.c
 * Description:  Float32 version of Conv2D with NCHW format (pre-padded input)
 *
 * Date:         23.01.2025
 *
 * ===================================================================== */

/*
 * Copyright (C) 2023 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Run Wang, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DeeployBasicMath.h"

void FusedConv2dRelu_fp32_fp32_fp32_NCHW(const float32_t *__restrict__ input, uint32_t C,
                                         uint32_t H_in, uint32_t W_in,
                                         uint32_t pad_bottom, uint32_t pad_left,
                                         uint32_t pad_top, uint32_t pad_right,
                                         const float32_t *__restrict__ weights,
                                         const float32_t *__restrict__ bias,
                                         uint32_t F, uint32_t P, uint32_t Q, uint32_t SP,
                                         uint32_t SQ, float32_t *__restrict__ output) {
  const uint32_t H_out = (H_in - P + pad_top + pad_bottom) / SP + 1;
  const uint32_t W_out = (W_in - Q + pad_left + pad_right) / SQ + 1;

  for (uint32_t f = 0; f < F; ++f) {
    for (uint32_t h = 0; h < H_out; ++h) {
      for (uint32_t w = 0; w < W_out; ++w) {
        float32_t sum = bias[f];
        for (uint32_t c = 0; c < C; ++c) {
          for (uint32_t p = 0; p < P; ++p) {
            for (uint32_t q = 0; q < Q; ++q) {
              const uint32_t w_in = w * SQ + q;
              const uint32_t h_in = h * SP + p;
              if (h_in >= pad_bottom && (pad_top + pad_bottom + H_in - h_in) > pad_top &&
                  w_in >= pad_left && (pad_left + pad_right + W_in - w_in) > pad_right) {
                sum += input[c * H_in * W_in + (h_in - pad_bottom) * W_in + (w_in - pad_left)] *
                       weights[f * C * P * Q + c * P * Q + p * Q + q];
              }
            }
          }
        }
        output[f * H_out * W_out + h * W_out + w] = sum > 0 ? sum : 0;
      }
    }
  }
}
