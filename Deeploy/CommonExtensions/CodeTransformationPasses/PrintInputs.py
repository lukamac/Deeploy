# ----------------------------------------------------------------------
#
# File: PrintInput.py
#
# Last edited: 13.11.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
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

import re
from typing import Callable, Literal, Tuple

from Deeploy.CommonExtensions.CodeTransformationPasses.IntrospectiveCodeTransformation import \
    IntrospectiveCodeTransformationMixIn
from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ConstantBuffer, ExecutionBlock, \
    NetworkContext, NodeTemplate, StructBuffer, TransientBuffer, VariableBuffer, _NoVerbosity

_DebugPrintTemplate = NodeTemplate("""
<%
accessStr = ""
dimStr = ""
for idx, dim in enumerate(bufferShape):
    accessStr += "[" + f"print_iter_{idx}" + "]"
    if idx > 0:
        dimStr += "[" + f"{dim}" + "]"
formatSpecifier = "%*i" 
if "float" in bufferType.referencedType.typeName or "double" in bufferType.referencedType.typeName:
    formatSpecifier = "%*.6f"  
%>
printf("${nodeName} ${bufferName}: ${bufferType.referencedType.typeName}, ${bufferShape}, %p\\n", ${bufferName});
% for idx, dim in enumerate(bufferShape):
printf("[");
for (int print_iter_${idx}=0; print_iter_${idx} < ${dim}; print_iter_${idx}++){
% endfor
printf("${formatSpecifier},", 4, ((${bufferType.referencedType.typeName} (*)${dimStr})${bufferName})${accessStr});
% for dim in bufferShape:
}
printf("], \\n");
%endfor
""")


class BufferPrintGeneration(CodeTransformationPass, IntrospectiveCodeTransformationMixIn):

    def __init__(self, addDirection: Literal["left", "right"], filter: Callable[[NetworkContext, VariableBuffer, str],
                                                                                bool]):
        self.addDirection = addDirection
        self.filter = filter

    def unrollReference(self, ctxt: NetworkContext, reference: str) -> VariableBuffer:
        buffer = ctxt.lookup(reference)
        while hasattr(buffer, "_referenceName"):
            buffer = ctxt.lookup(buffer._referenceName)
        return buffer

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        references = self.extractDynamicReferences(ctxt,
                                                   executionBlock,
                                                   unrollStructs = True,
                                                   includeGobalReferences = True)

        for ref in references:
            buffer = self.unrollReference(ctxt, ref)
            if self.filter(ctxt, buffer, ref):
                operatorRepresentation = {
                    "bufferName": ref,
                    "bufferType": buffer._type,
                    "bufferShape": buffer.shape,
                    "nodeName": name,
                }
                if self.addDirection == "left":
                    executionBlock.addLeft(_DebugPrintTemplate, operatorRepresentation)
                elif self.addDirection == "right":
                    executionBlock.addRight(_DebugPrintTemplate, operatorRepresentation)
                else:
                    raise RuntimeError(f"Unrecognized addDirection {self.addDirection}")

        return ctxt, executionBlock


class PrintInputGeneration(BufferPrintGeneration):

    def __init__(self):

        def filter(ctxt: NetworkContext, buffer: VariableBuffer, nodeName: str) -> bool:
            return all([
                not isinstance(buffer, (TransientBuffer, ConstantBuffer, StructBuffer)),
                nodeName in buffer._users,
            ])

        super().__init__("left", filter)


class PrintOutputGeneration(BufferPrintGeneration):

    def __init__(self):

        def filter(ctxt: NetworkContext, buffer: VariableBuffer, nodeName: str) -> bool:
            return all([
                not isinstance(buffer, (TransientBuffer, ConstantBuffer, StructBuffer)),
                nodeName in buffer._users,
                len(buffer._users) > 0 or ctxt.is_global(buffer.name),
            ])

        super().__init__("right", filter)


class PrintConstantGeneration(BufferPrintGeneration):

    def __init__(self):

        def filter(ctxt: NetworkContext, buffer: VariableBuffer, nodeName: str) -> bool:
            return isinstance(buffer, ConstantBuffer) and len(buffer._users) > 0

        super().__init__("left", filter)


class MemoryUnawareMixIn():

    def memoryUnawareFilter(self):

        def newFilter(ctxt: NetworkContext, buffer: VariableBuffer, nodeName: str) -> bool:
            return not hasattr(buffer, "_memoryLevel") and self.filter(ctxt, buffer, nodeName)


class MemoryAwareMixIn():

    def __init__(self, regex: str) -> None:
        self.regex = re.compile(regex)

    def memoryAwareFilter(self):

        def newFilter(ctxt: NetworkContext, buffer: VariableBuffer, nodeName: str) -> bool:
            return hasattr(buffer, "_memoryLevel") and self.regex.fullmatch(
                buffer._memoryLevel) is not None and self.filter(ctxt, buffer, nodeName)


class MemoryAwarePrintInputGeneration(PrintInputGeneration, MemoryAwareMixIn):

    def __init__(self, memoryLevelRegex: str):
        super().__init__()
        MemoryAwareMixIn.__init__(self, memoryLevelRegex)
        self.filter = self.memoryAwareFilter()


class MemoryUnawarePrintInputGeneration(PrintInputGeneration, MemoryUnawareMixIn):

    def __init__(self):
        super().__init__()
        self.filter = self.memoryUnawareFilter()


class MemoryAwarePrintOutputGeneration(PrintOutputGeneration, MemoryAwareMixIn):

    def __init__(self, memoryLevelRegex: str):
        super().__init__()
        MemoryAwareMixIn.__init__(self, memoryLevelRegex)
        self.filter = self.memoryAwareFilter()


class MemoryUnawarePrintOutputGeneration(PrintOutputGeneration, MemoryUnawareMixIn):

    def __init__(self):
        super().__init__()
        self.filter = self.memoryUnawareFilter()


class MemoryAwarePrintConstantGeneration(PrintConstantGeneration, MemoryAwareMixIn):

    def __init__(self, memoryLevelRegex: str):
        super().__init__()
        MemoryAwareMixIn.__init__(self, memoryLevelRegex)
        self.filter = self.memoryAwareFilter()


class MemoryUnawarePrintConstantGeneration(PrintConstantGeneration, MemoryUnawareMixIn):

    def __init__(self):
        super().__init__()
        self.filter = self.memoryUnawareFilter()
