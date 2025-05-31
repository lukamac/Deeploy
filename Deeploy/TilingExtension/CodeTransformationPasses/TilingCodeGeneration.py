# ----------------------------------------------------------------------
#
# File: TilingCodeGeneration.py
#
# Last edited: 24.10.2023
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

from abc import abstractmethod
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar

import Deeploy.CommonExtensions.DataTypes as BasicDataTypes
from Deeploy.AbstractDataTypes import Immediate, PointerClass
from Deeploy.CommonExtensions.CodeTransformationPasses.Closure import ClosureExecutionBlock
from Deeploy.CommonExtensions.CodeTransformationPasses.IntrospectiveCodeTransformation import \
    IntrospectiveCodeTransformationMixIn
from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import ArgumentStructGeneration
from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ConstantBuffer, ExecutionBlock, \
    NetworkContext, NodeTemplate, OperatorRepresentation, _NoVerbosity
from Deeploy.TilingExtension.CodeTransformationPasses.TilingPrototypes import PrototypeTilingMixIn
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint, TensorMemoryConstraint
from Deeploy.TilingExtension.TilingCodegen import TilingSchedule, VariableReplacementScheme, minimizeVariableReplacement

KT = TypeVar('KT')
VT = TypeVar('VT')


def dictOfArrays(arrayOfDicts: Sequence[Mapping[KT, VT]]) -> Mapping[KT, List[VT]]:
    ret: Mapping[KT, List[VT]] = {}
    for i, _dict in enumerate(arrayOfDicts):
        if i == 0:
            ret.update({key: [value] for key, value in _dict.items()})
        else:
            assert set(ret.keys()) == set(_dict.keys()), "Keys should be the same"
            for key, value in _dict.items():
                ret[key].append(value)
    return ret


T = TypeVar('T')


def transposeListOfLists(listOfLists: List[List[T]]) -> List[List[T]]:
    transposedListOfLists = []
    for _list in listOfLists:
        for i, element in enumerate(_list):
            if i >= len(transposedListOfLists):
                assert i == len(transposedListOfLists)
                transposedListOfLists.append([element])
            else:
                transposedListOfLists[i].append(element)
    return transposedListOfLists


class TilingCodeGeneration(CodeTransformationPass, IntrospectiveCodeTransformationMixIn, PrototypeTilingMixIn):

    def __init__(self, targetMemLevel: str):
        self.targetMemLevel = targetMemLevel
        self.argStructGeneration = ArgumentStructGeneration()

    @abstractmethod
    def generateTilingLoop(
            self, ctxt: NetworkContext, executionBlock: ExecutionBlock, nodeMemoryConstraint: NodeMemoryConstraint,
            tilingSchedule: TilingSchedule, variableReplacement: VariableReplacementScheme,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, ExecutionBlock, bool]:

        return ctxt, executionBlock, False

    # SCHEREMO: internalPtr refers to the HIGHER memory level of a transfer,
    # e.g. in both an L2 -> L1 and L1 -> L2 transfer, the internalPtr is in L1.
    @staticmethod
    def isFinalMemoryLevel(tensorMemoryConstraint: TensorMemoryConstraint, memory: str) -> bool:
        memoryOrder = list(tensorMemoryConstraint.memoryConstraints.keys())
        assert memory in memoryOrder, f"Memory {memory} does not exist in the tensor memory constraint {tensorMemoryConstraint}"
        if len(memoryOrder) < 2:
            return True
        return memory in memoryOrder[:2]

    def _hoistTileIdxPtr(self,
                         ctxt: NetworkContext,
                         operatorRepresentation: OperatorRepresentation,
                         sourceMemoryLevel: str = "L2") -> str:

        newPtrName = self.prefix + operatorRepresentation['nodeName'] + "_tileIdxPtr"

        tilePtrBuffer = ctxt.VariableBuffer(newPtrName, shape = [1])
        ctxt.add(tilePtrBuffer, "local")

        _type = ctxt.lookup(self.prefix + operatorRepresentation['nodeName'] + "_numTiles")._type

        tilePtrBuffer._type = _type
        tilePtrBuffer._instance = tilePtrBuffer._type(newPtrName, ctxt)
        tilePtrBuffer._memoryLevel = sourceMemoryLevel

        tilePtrBuffer.allocTemplate = NodeTemplate("")
        tilePtrBuffer.deallocTemplate = NodeTemplate("")
        tilePtrBuffer.initTemplate = NodeTemplate("""
        ${type.referencedType.typeName} bu_${name} = 0;
        ${type.referencedType.typeName}* ${name} = &bu_${name};""")

        return newPtrName

    def _hoistValues(self, ctxt: NetworkContext, name: str, values: List[int], nodeName: str) -> ConstantBuffer:
        cb = ctxt.ConstantBuffer(name, [len(values)], values)
        ctxt.add(cb, 'global')
        cb._type = PointerClass(BasicDataTypes.minimalIntegerType(values))
        cb._instance = cb._type(cb.name, ctxt)
        cb._memoryLevel = self.targetMemLevel
        cb._users.append(nodeName)
        return cb

    def _hoistNumTiles(self,
                       ctxt: NetworkContext,
                       nodeName: str,
                       tilingSchedules: List[TilingSchedule],
                       sourceMemoryLevel: str = "L2") -> str:
        stepsNumTiles = [len(tilingSchedule.outputLoadSchedule) for tilingSchedule in tilingSchedules]

        cumulativeNumTiles = [0]
        for numTiles in stepsNumTiles:
            cumulativeNumTiles.append(cumulativeNumTiles[-1] + numTiles)

        cb = self._hoistValues(ctxt, f"{self.prefix}{nodeName}_numTiles", cumulativeNumTiles, nodeName)

        return cb.name

    def _hoistOpReprUpdates(self,
                            ctxt: NetworkContext,
                            opReprs: List[OperatorRepresentation],
                            nodeName: str,
                            prefix: str = '') -> OperatorRepresentation:
        # Early exit if the opReprs list is empty because the following code assumes at least 1 opRepr is in the list
        if len(opReprs) == 0:
            return {}

        newOpRepr = {}
        for var, updates in dictOfArrays(opReprs).items():
            if all(update == updates[0] for update in updates):
                newOpRepr[var] = updates[0]
            elif isinstance(updates[0], (list, tuple)):
                newVarList = []
                for i, values in enumerate(transposeListOfLists(updates)):
                    if all(value == values[0] for value in values):
                        newVarList.append(values[0])
                    else:
                        cb = self._hoistValues(ctxt, f"{prefix}{var}_{i}", values, nodeName)
                        newVarList.append(cb.name)
                newOpRepr[var] = newVarList
            else:
                cb = self._hoistValues(ctxt, f"{prefix}{var}", updates, nodeName)
                newOpRepr[var] = cb.name
        return newOpRepr

    def _hoistConstantAndReference(self,
                                   ctxt: NetworkContext,
                                   constBuf: ConstantBuffer,
                                   operatorRepresentation: OperatorRepresentation,
                                   nodeName: str,
                                   operatorRepresentationName: str,
                                   immediateType: Optional[Type[Immediate]] = None) -> Tuple[NetworkContext, Dict]:

        if immediateType is None:
            _type = PointerClass(BasicDataTypes.int32_t)
        else:
            _type = PointerClass(immediateType)

        name = constBuf.name

        ctxt.add(constBuf, "global")
        constBuf._type = _type
        constBuf._instance = constBuf._type(name, ctxt)
        constBuf._users = [nodeName]
        constBuf._memoryLevel = self.targetMemLevel

        ref = ctxt.hoistReference(name + "_ref", constBuf)
        ref._memoryLevel = self.targetMemLevel

        operatorRepresentation[operatorRepresentationName] = ref.name

        return ctxt, operatorRepresentation

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:

        def unravelReference(ctxt: NetworkContext, name: str) -> str:

            if name not in ctxt.localObjects.keys() and name not in ctxt.globalObjects.keys():
                return name

            refBuffer = ctxt.lookup(name)
            if not hasattr(refBuffer, "_referenceName"):
                return name

            return unravelReference(ctxt, refBuffer._referenceName)

        if isinstance(executionBlock, ClosureExecutionBlock):
            baseExecutionBlock = executionBlock.baseBlock
        else:
            baseExecutionBlock = executionBlock

        patternMemoryConstraint = baseExecutionBlock.patternMemoryConstraint

        if patternMemoryConstraint is None:
            return ctxt, executionBlock

        assert len(patternMemoryConstraint.nodeConstraints) == 1, "Only layerwise supported for now!"
        #assert len(baseExecutionBlock.codeSnippets) == 1, "Only layerwise supported for now!"

        nodeMemoryConstraint = patternMemoryConstraint.nodeConstraints[0]

        possibleTemplateNodes = [
            node for node in baseExecutionBlock.codeSnippets if hasattr(node.template, 'tileConstraint')
        ]

        assert len(possibleTemplateNodes) == 1, "More than one template node with TCF found"

        templateNode = possibleTemplateNodes[0]

        operatorRepresentation = templateNode.operatorRepresentation
        unravelRep = operatorRepresentation.copy()
        for key in unravelRep.keys():

            val = unravelRep[key]
            if not isinstance(val, str):
                continue

            unravelRep[key] = unravelReference(ctxt, val)

        template = templateNode.template

        variableReplacement, tilingSchedules = template.tileConstraint.wrapTilingSolution(
            nodeMemoryConstraint, self.targetMemLevel, ctxt, unravelRep)

        minimalVariableReplacement, newNodeRep = minimizeVariableReplacement(variableReplacement,
                                                                             templateNode.operatorRepresentation)
        for key, value in newNodeRep.items():
            templateNode.operatorRepresentation[key] = value

        ctxt, executionBlock, applicable = self.generateTilingLoop(ctxt, executionBlock, nodeMemoryConstraint,
                                                                   tilingSchedules, minimalVariableReplacement,
                                                                   operatorRepresentation)
        if applicable:
            ctxt, executionBlock = self.argStructGeneration.apply(ctxt, executionBlock, name)

        return ctxt, executionBlock
