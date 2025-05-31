# ----------------------------------------------------------------------
#
# File: TilingVariableReplacement.py
#
# Last edited: 28.09.2023
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

import copy
from typing import Any, Dict, List, Tuple, Type

from mako.parsetree import Expression, Node, Text

from Deeploy.AbstractDataTypes import Pointer, Struct
from Deeploy.CommonExtensions.CodeTransformationPasses.Closure import ClosureExecutionBlock
from Deeploy.CommonExtensions.CodeTransformationPasses.IntrospectiveCodeTransformation import \
    IntrospectiveCodeTransformationMixIn
from Deeploy.DeeployTypes import CodeGenVerbosity, CodeSnippet, CodeTransformationPass, ExecutionBlock, \
    NetworkContext, NodeTemplate, OperatorRepresentation, TransientBuffer, VariableBuffer, _NoVerbosity, \
    _ReferenceBuffer
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TilingCodegen import TilingSchedule, VariableReplacementScheme, minimizeVariableReplacement


class TilingVariableReplacement(CodeTransformationPass, IntrospectiveCodeTransformationMixIn):

    _prefix = "TILING_REPLACED_"

    def __init__(self, targetMemLevel: str):
        self.targetMemLevel = targetMemLevel
        self._name: str

    @property
    def prefix(self):
        return self._prefix + f"{self._name}_" + self.targetMemLevel + "_"

    @property
    def arenaName(self):
        return f"MEMORYARENA_{self.targetMemLevel}"

    def _arenaAllocate(self, ctxt: NetworkContext, buffer: VariableBuffer, offset: int) -> VariableBuffer:
        arena = ctxt.lookup(self.arenaName)
        buffer.allocTemplate = NodeTemplate(" \
        ${type.typeName} ${name} = (${type.typeName}) " + f"((char*){str(arena._instance)} + {offset});")
        buffer.deallocTemplate = NodeTemplate("")
        return buffer

    def _dereferencePointer(self, nodes: List[Node], name: str) -> List[Node]:
        instanceIdxs = [idx for idx, node in enumerate(nodes) if isinstance(node, Expression) and node.text == name]

        for offset, idx in enumerate(instanceIdxs):
            text = Text("*", source = "*", lineno = 0, pos = 0, filename = None)
            nodes.insert(offset + idx, text)

        return nodes

    def _hoistAndReferenceValues(self, ctxt: NetworkContext, name: str, values: List[int],
                                 _type: Type[Pointer]) -> _ReferenceBuffer:
        cb = ctxt.ConstantBuffer(name, (len(values),), values)
        cb._type = _type
        ctxt.add(cb, 'global')
        cb._instance = cb._type(cb.name, ctxt)
        cb._memoryLevel = self.targetMemLevel

        ref = ctxt.hoistReference(name + "_ref", cb)
        ref._memoryLevel = self.targetMemLevel
        return ref

    def _replaceReference(self, ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation,
                          tilingSchedule: TilingSchedule, name: str) -> Tuple[NetworkContext, Dict]:
        if name in tilingSchedule.inputBaseOffsets:
            offset = tilingSchedule.inputBaseOffsets[name][0]
        elif name in tilingSchedule.outputBaseOffsets:
            offset = tilingSchedule.outputBaseOffsets[name][0]
        else:
            raise RuntimeError(f"Name {name} not found in TilingSchedule {tilingSchedule}")

        buffer = ctxt.lookup(operatorRepresentation[name])
        unraveledBuffer = ctxt.unravelReference(buffer)

        ref = ctxt.hoistReference(self.prefix + name + "_ref", unraveledBuffer)
        ref._memoryLevel = self.targetMemLevel
        ref = self._arenaAllocate(ctxt, ref, offset)
        operatorRepresentation[name] = ref.name

        return ctxt, operatorRepresentation

    def _replaceTransients(self, ctxt: NetworkContext, tensorName: str,
                           nodeMemoryConstraint: NodeMemoryConstraint) -> NetworkContext:
        memoryConstraints = nodeMemoryConstraint.tensorMemoryConstraints[tensorName].memoryConstraints
        assert len(memoryConstraints) == 1, f"Tiled transient buffer {tensorName} has more than one memory level!"
        key = list(memoryConstraints.keys())[0]
        constraint = memoryConstraints[key]
        assert constraint.addrSpace is not None, f"Address space of {constraint} cannot be None!"
        offset = constraint.addrSpace[0]

        ref = ctxt.lookup(tensorName)

        if ref._memoryLevel != self.targetMemLevel:
            return ctxt

        _ = self._arenaAllocate(ctxt, ref, offset)
        return ctxt

    def _replaceTiledExpressions(self, ctxt: NetworkContext, snippet: CodeSnippet,
                                 variableReplacement: VariableReplacementScheme, tilingSchedule: TilingSchedule,
                                 nodeMemoryConstraint: NodeMemoryConstraint) -> NetworkContext:

        operatorRepresentation = snippet.operatorRepresentation
        template = snippet.template

        inoutSchedule = {**tilingSchedule.inputBaseOffsets, **tilingSchedule.outputBaseOffsets}
        variableList = [key for key in inoutSchedule.keys() if type(operatorRepresentation[key]) == str]

        transientBufferList = []
        for key, value in operatorRepresentation.items():
            if not isinstance(value, str):
                continue
            if (ctxt.is_local(value) and isinstance(ctxt.lookup(value), TransientBuffer)):
                transientBufferList.append(key)

        parseTree = IntrospectiveCodeTransformationMixIn._generateParseTree(template)
        newParseTree = copy.copy(parseTree)
        nodes = parseTree.nodes
        newNodes = copy.copy(nodes)

        for varName, varReplacementValues in variableReplacement.perTileReplacements.items():
            if isinstance(operatorRepresentation[varName], str):
                continue
            ref = self._hoistAndReferenceValues(ctxt, self.prefix + varName, varReplacementValues,
                                                variableReplacement.replacementTypes[varName])
            operatorRepresentation[varName] = ref.name
            newNodes = self._dereferencePointer(newNodes, varName)

        for rep in variableList:
            ctxt, operatorRepresentation = self._replaceReference(ctxt, operatorRepresentation, tilingSchedule, rep)

        for rep in transientBufferList:
            ctxt = self._replaceTransients(ctxt, operatorRepresentation[rep], nodeMemoryConstraint)

        newParseTree.nodes = newNodes
        IntrospectiveCodeTransformationMixIn._reconstructCode(template, newParseTree)

        return ctxt

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        self._name = name

        if isinstance(executionBlock, ClosureExecutionBlock):
            baseExecutionBlock = executionBlock.baseBlock
        else:
            baseExecutionBlock = executionBlock

        patternMemoryConstraint = baseExecutionBlock.patternMemoryConstraint

        if patternMemoryConstraint is None:
            return ctxt, executionBlock

        assert len(patternMemoryConstraint.nodeConstraints) == 1, "Only layerwise supported for now!"
        #assert len(executionBlock.codeSnippets) == 1, "Only layerwise supported for now!"

        nodeMemoryConstraint = patternMemoryConstraint.nodeConstraints[0]

        possibleTemplateNodes = [
            node for node in baseExecutionBlock.codeSnippets if hasattr(node.template, 'tileConstraint')
        ]

        assert len(possibleTemplateNodes) == 1, "More than one template node with TCF found"

        templateNode = possibleTemplateNodes[0]
        operatorRepresentation = templateNode.operatorRepresentation
        template = templateNode.template

        def is_buffer(value: Any) -> bool:
            return isinstance(value, str) and (ctxt.is_local(value) or ctxt.is_global(value))

        unraveledOpRepr = {
            key: ctxt.unravelReference(ctxt.lookup(value)).name if is_buffer(value) else value
            for key, value in operatorRepresentation.items()
        }

        variableReplacement, tilingSchedules = template.tileConstraint.wrapTilingSolution(
            nodeMemoryConstraint, self.targetMemLevel, ctxt, unraveledOpRepr)

        minimalVariableReplacement, newOpRepr = minimizeVariableReplacement(variableReplacement, operatorRepresentation)
        for key, value in newOpRepr.items():
            templateNode.operatorRepresentation[key] = value

        flatTilingSchedule = copy.copy(tilingSchedules[0])
        for tilingSchedule in tilingSchedules[1:]:
            flatTilingSchedule += tilingSchedule

        ctxt = self._replaceTiledExpressions(ctxt, templateNode, minimalVariableReplacement, flatTilingSchedule,
                                             nodeMemoryConstraint)

        tilingReplacedRefMap = {}
        for key in list(flatTilingSchedule.inputBaseOffsets.keys()) + list(flatTilingSchedule.outputBaseOffsets.keys()):
            tilingReplacedRefMap[unraveledOpRepr[key]] = operatorRepresentation[key]

        # Swap any original tensor occurances with the tiled targetMemLevel-local tensor
        for codeSnippet in executionBlock.codeSnippets:
            template, opRepr = codeSnippet.template, codeSnippet.operatorRepresentation

            for key, value in opRepr.items():
                if isinstance(value, str) and value in tilingReplacedRefMap:
                    opRepr[key] = tilingReplacedRefMap[value]

            if "closureStructArgs" in opRepr:
                closureArgsStruct: Struct = opRepr['closureStructArgs']
                structDict = closureArgsStruct.value

                for key, value in structDict.items():
                    if value.referenceName in tilingReplacedRefMap:
                        structDict[key] = type(value)(tilingReplacedRefMap[value.referenceName], ctxt)

        return ctxt, executionBlock
