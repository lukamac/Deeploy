# ----------------------------------------------------------------------
#
# File: FLoatConvTemplate.py
#
# Last edited: 12.05.2025
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Authors:
# - Run Wang, ETH Zurich
# - Calin Diaconu, University of Bologna
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

from Deeploy.Targets.Generic.Templates.FloatConvTemplate import FloatConv2dTemplate

parallelTemplate = FloatConv2dTemplate("""
<%
quotient = f"{nodeName}_F_quotient"
remainder = f"{nodeName}_F_remainder"
begin = f"{nodeName}_F_begin"
end = f"{nodeName}_F_end"

batchOffsetIn = ch_im_in * dim_im_in_x * dim_im_in_y
batchOffsetOut = ch_im_out * dim_im_out_x * dim_im_out_y
%>
// 2D FP Conv (Name: ${nodeName}, Op: ${nodeOp})
const uint32_t ${quotient} = ${ch_im_out} / numThreads;
const uint32_t ${remainder} = ${ch_im_out} % numThreads;

const uint32_t ${begin} = ${quotient} * core_id + (core_id < ${remainder} ? core_id : ${remainder});
const uint32_t ${end} = ${begin} + ${quotient} + (core_id < ${remainder} ? 1 : 0);

${data_in_type.typeName} ${data_in_ref} = ${data_in};
${data_out_type.typeName} ${data_out_ref} = ${data_out};

for (uint32_t n=0; n<${batch}; ++n) {
    Conv2d_fp${data_in_type.referencedType.typeWidth}_fp${weight_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}_NCHW(
        ${data_in_ref}, ${ch_im_in}, ${dim_im_in_x}, ${dim_im_in_y},
        ${weight}, ${begin}, ${end}, ${dim_kernel_x}, ${dim_kernel_y},
        ${stride_x}, ${stride_y},
        ${bias},
        ${has_bias},
        ${data_out_ref}
    );
    ${data_in_ref} += ${batchOffsetIn};
    ${data_out_ref} += ${batchOffsetOut};
}
""")
