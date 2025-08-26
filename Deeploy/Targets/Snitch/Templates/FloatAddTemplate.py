# ----------------------------------------------------------------------
#
# File: FloatAddTemplate.py
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

from typing import List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer

referenceTemplate = NodeTemplate("""
// Snitch Float Add (Name: ${nodeName}, Op: ${nodeOp})
SnitchFloatAdd(${data_in_1}, ${data_in_2}, ${data_out}, ${size});
""")


class FloatAddTemplate(NodeTemplate):

    def alignToContext(
            self, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, OperatorRepresentation, List[str]]:
        buff = ctxt.lookup(operatorRepresentation["data_out"])
        assert isinstance(buff, VariableBuffer)
        operatorRepresentation["out_type"] = buff._type.referencedType.typeName
        return ctxt, operatorRepresentation, []


singleCoreFloatAddTemplate = FloatAddTemplate("""
// Snitch Float Add (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
for (uint32_t i = 0; i < ${size}; i++) {
    ${data_out}[i] = (${out_type})${data_in_1}[i] + (${out_type})${data_in_2}[i];
}
END_SINGLE_CORE
""")

multiCoreFloatAddTemplate = FloatAddTemplate("""
// Snitch Float Add (Name: ${nodeName}, Op: ${nodeOp})
const uint32_t ${nodeName}_quotient = ${size} / numThreads;
const uint32_t ${nodeName}_remainder = ${size} % numThreads;

const uint32_t ${nodeName}_begin = ${nodeName}_quotient * core_id + (core_id < ${nodeName}_remainder ? core_id : ${nodeName}_remainder);
const uint32_t ${nodeName}_end = ${nodeName}_begin + ${nodeName}_quotient + (core_id < ${nodeName}_remainder ? 1 : 0);

for (uint32_t i = ${nodeName}_begin; i < ${nodeName}_end; i++) {
    ${data_out}[i] = (${out_type})${data_in_1}[i] + (${out_type})${data_in_2}[i];
}
""")
