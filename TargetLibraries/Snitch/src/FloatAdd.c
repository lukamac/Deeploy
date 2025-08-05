/* ----------------------------------------------------------------------
#
# File: FloatAdd.c
#
# Last edited: 11.06.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author:
# - Luka Macan, luka.macan@unibo.it, University of Bologna
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
*/

#include "DeeploySnitchMath.h"

void SnitchFloatAdd(float32_t *pIn1, float32_t *pIn2, float32_t *pOut, uint32_t size) {
  const uint32_t core_id = snrt_global_compute_core_idx();
  const uint32_t numThreads = snrt_global_compute_core_num();

  const uint32_t quotient = size / numThreads;
  const uint32_t remainder = size % numThreads;

  const uint32_t chunk_size = core_id < remainder ? quotient + 1 : quotient;
  const uint32_t chunk_begin = quotient * core_id + (core_id < remainder ? core_id : remainder);
  const uint32_t chunk_end = chunk_begin + chunk_size;

#pragma loopunroll 2
  for (int i = chunk_begin; i < chunk_end; i++) {
    pOut[i] = pIn1[i] + pIn2[i];
  }
}
