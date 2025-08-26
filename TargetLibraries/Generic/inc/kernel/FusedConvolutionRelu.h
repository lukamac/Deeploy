/* =====================================================================
 * Title:        Convolution.h
 * Description:
 *
 * Date:         04.01.2023
 *
 * ===================================================================== */

/*
 * Copyright (C) 2023 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Philip Wiese, ETH Zurich
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
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __DEEPLOY_BASIC_MATH_FUSED_CONVOLUTION_RELU_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_FUSED_CONVOLUTION_RELU_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/* This file implements fused convolution and relu.
 *
 * A is an M x N input matrix, B is a P x Q kernel matrix and C is and M x N
 * output matrix
 *
 */

/******************************************************************************/
/*                         Fused Convolution Relu                             */
/******************************************************************************/

/*
 * 2D Convolution  ----------------------------------
 * kernel      = FusedConv2dRelu_fp32_fp32_fp32_NCHW
 * layout      = NCHW
 * data type   = 32-bit float
 * kernel size = generic
 * unrolling   = no
 * simd        = no
 * potentially parallelizable = yes
 */
void FusedConv2dRelu_fp32_fp32_fp32_NCHW(const float32_t *__restrict__ input, uint32_t C,
                                         uint32_t H_padded, uint32_t W_padded,
                                         uint32_t pad_bottom, uint32_t pad_left,
                                         uint32_t pad_top, uint32_t pad_right,
                                         const float32_t *__restrict__ weights,
                                         const float32_t *__restrict__ bias,
                                         uint32_t F_begin, uint32_t F_end, uint32_t P, uint32_t Q,
                                         uint32_t SP, uint32_t SQ, float32_t *__restrict__ output);

#endif //__DEEPLOY_BASIC_MATH_FUSED_CONVOLUTION_RELU_KERNEL_HEADER_
