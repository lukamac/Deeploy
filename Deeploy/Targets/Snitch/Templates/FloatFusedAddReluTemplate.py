# ----------------------------------------------------------------------
#
# File: FloatFusedAddReluTemplate.py
#
# Last edited: 13.11.2024
#
# Copyright (C) 2021, ETH Zurich and University of Bologna.
#
# Authors:
# - Francesco Conti, UNIBO
# - Alberto Dequino, UNIBO
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// FusedAddRelu (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
for (uint32_t i=0; i < ${size}; i++){
    const float res = ${data_in_1}[i] + ${data_in_2}[i];
    ${data_out}[i] = res > 0.0f ? res : 0.0f;
}
END_SINGLE_CORE
""")

superFastParallelTemplate = NodeTemplate("""
<%
quotient = f"{nodeName}_size_quotient"
remainder = f"{nodeName}_size_remainder"
begin = f"{nodeName}_size_begin"
end = f"{nodeName}_size_end"
%>
// FusedAddRelu (Name: ${nodeName}, Op: ${nodeOp})
const uint32_t ${quotient} = ${size} / numThreads;
const uint32_t ${remainder} = ${size} % numThreads;

const uint32_t ${begin} = ${quotient} * core_id + (core_id < ${remainder} ? core_id : ${remainder});
const uint32_t ${end} = ${begin} + ${quotient} + (core_id < ${remainder} ? 1 : 0);

for (uint32_t i=${begin}; i < ${end}; i++){
    const float res = ${data_in_1}[i] + ${data_in_2}[i];
    ${data_out}[i] = res > 0.0 ? res : 0.0;
}
""")
