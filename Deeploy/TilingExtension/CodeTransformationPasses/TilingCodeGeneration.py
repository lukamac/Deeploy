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

import copy
from abc import abstractmethod
from typing import List, Tuple, TypeVar

import numpy as np

from Deeploy.CommonExtensions.CodeTransformationPasses.Closure import ClosureExecutionBlock
from Deeploy.CommonExtensions.CodeTransformationPasses.IntrospectiveCodeTransformation import \
    IntrospectiveCodeTransformationMixIn
from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import ArgumentStructGeneration
from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, \
    NodeTemplate, OperatorRepresentation, VariableBuffer, _NoVerbosity
from Deeploy.TilingExtension.CodeTransformationPasses.TilingHoistingMixIn import TilingHoistingMixIn
from Deeploy.TilingExtension.CodeTransformationPasses.TilingPrototypes import PrototypeTilingMixIn
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint, TensorMemoryConstraint
from Deeploy.TilingExtension.TilingCodegen import HyperRectangle, TilingSchedule, VariableReplacementScheme, \
    minimizeRectangle, minimizeVariableReplacement

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


class TilingCodeGeneration(CodeTransformationPass, IntrospectiveCodeTransformationMixIn, PrototypeTilingMixIn,
                           TilingHoistingMixIn):

    @abstractmethod
    def generateTilingLoop(
            self, ctxt: NetworkContext, executionBlock: ExecutionBlock, nodeMemoryConstraint: NodeMemoryConstraint,
            tilingSchedules: List[TilingSchedule], variableReplacement: VariableReplacementScheme,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, ExecutionBlock, bool]:

        return ctxt, executionBlock, False

    def __init__(self, targetMemLevel: str):
        self.targetMemLevel = targetMemLevel
        self.argStructGeneration = ArgumentStructGeneration()
        TilingHoistingMixIn.__init__(self, targetMemLevel)

    # SCHEREMO: internalPtr refers to the HIGHER memory level of a transfer,
    # e.g. in both an L2 -> L1 and L1 -> L2 transfer, the internalPtr is in L1.
    def isFinalMemoryLevel(self, tensorMemoryConstraint: TensorMemoryConstraint) -> bool:
        memoryOrder = list(tensorMemoryConstraint.memoryConstraints.keys())
        assert self.targetMemLevel in memoryOrder, f"Memory {self.targetMemLevel} does not exist in the tensor memory constraint {tensorMemoryConstraint}"
        if len(memoryOrder) < 2:
            return True
        return self.targetMemLevel in memoryOrder[:2]

    @staticmethod
    def padShape(shape: Tuple[int, ...], rank: int) -> Tuple[int, ...]:
        assert rank >= len(
            shape), f"Cannot pad to rank smaller then shape's. Received rank: {rank}, shape rank: {len(shape)}"
        ret = tuple([1] * (rank - len(shape))) + shape
        assert len(ret) == rank
        return ret

    @staticmethod
    def padOffset(offset: Tuple[int, ...], rank: int) -> Tuple[int, ...]:
        assert rank >= len(
            offset), f"Cannot pad to rank smaller then offset's. Received rank: {rank}, offset rank: {len(offset)}"
        ret = tuple([0] * (rank - len(offset))) + offset
        assert len(ret) == rank
        return ret

    @staticmethod
    def padStride(stride: Tuple[int, ...], rank: int, paddingStride: int) -> Tuple[int, ...]:
        assert rank >= len(
            stride), f"Cannot pad to rank smaller then stride's. Received rank: {rank}, stride rank: {len(stride)}"
        ret = tuple([paddingStride] * (rank - len(stride))) + stride
        assert len(ret) == rank
        return ret

    # TODO: Not super sure this should go here. It could be shared, but it seems a little bit too specific
    # with the `isFinalMemory` thing.
    def _legalizeTransfers(self, transfers: List[HyperRectangle], outerShape: Tuple[int, ...], typeWidth: int,
                           isFinalMemoryLevel: bool) -> Tuple[List[HyperRectangle], Tuple[int, ...]]:
        transfersCommonRank = max(len(rect.dims) for rect in transfers)
        commonRank = max(transfersCommonRank, len(outerShape))
        outerShape = self.padShape(outerShape, commonRank)

        minOuterShape = None

        if isFinalMemoryLevel:
            minimizedTransfers = []
            for rect in transfers:
                paddedRect = HyperRectangle(self.padOffset(rect.offset, commonRank),
                                            self.padShape(rect.dims, commonRank))
                minRect, newMinOuterShape = minimizeRectangle(paddedRect, outerShape)
                if minOuterShape is None:
                    minOuterShape = newMinOuterShape
                else:
                    if minOuterShape != newMinOuterShape:
                        rectStr = "\n".join(str(trans) for trans in transfers[:transfers.index(rect)])
                        raise RuntimeError(f"""Currently support a single minimal outer shape.
Old minOuterShape: {minOuterShape} vs. new minOuterShape {newMinOuterShape}.
New minOuterShape produced by outerDims: {outerShape} and rect: {rect}.
Old minOuterShape produced by outerDims: {outerShape} and rects:
{rectStr}""")
                minimizedTransfers.append(minRect)
        else:
            minimizedTransfers = [HyperRectangle((0,), (int(np.prod(rect.dims)),)) for rect in transfers]
            minOuterShape = (int(np.prod(outerShape)),)

        if minOuterShape is not None:
            outerShape = minOuterShape
        transfers = minimizedTransfers

        def sizeInBytes(length: int, typeWidth: int) -> int:
            return int(np.ceil((length * typeWidth) / 8))

        outerShape = outerShape[:-1] + (sizeInBytes(outerShape[-1], typeWidth),)

        inBytesTransfers = []
        for rect in transfers:
            newOffset = rect.offset[:-1] + (sizeInBytes(rect.offset[-1], typeWidth),)
            newDims = rect.dims[:-1] + (sizeInBytes(rect.dims[-1], typeWidth),)
            inBytesTransfers.append(HyperRectangle(newOffset, newDims))
        transfers = inBytesTransfers

        return transfers, outerShape

    def _tileTemplate(self, ctxt: NetworkContext, perTileOpReprs: List[OperatorRepresentation], template: NodeTemplate,
                      tileIdxVar: str, prefix: str) -> Tuple[NodeTemplate, OperatorRepresentation]:
        opRepr, hoistedNames = self._hoistOpReprUpdates(ctxt, perTileOpReprs, prefix)
        if len(hoistedNames) > 0:
            template = copy.deepcopy(template)
            self.indexVars(template.template, hoistedNames, "tileIdxVar")
            opRepr["tileIdxVar"] = tileIdxVar
        return template, opRepr

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
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

        self._initPrefix(templateNode.operatorRepresentation['nodeName'])

        operatorRepresentation = templateNode.operatorRepresentation

        unraveledOpRepr = operatorRepresentation.copy()
        for key, value in unraveledOpRepr.items():
            if ctxt.is_buffer(value):
                buffer = ctxt.lookup(value)
                assert isinstance(buffer, VariableBuffer)
                unraveledOpRepr[key] = ctxt.unravelReference(buffer).name

        template = templateNode.template

        variableReplacement, tilingSchedules = template.tileConstraint.wrapTilingSolution(
            nodeMemoryConstraint, self.targetMemLevel, ctxt, unraveledOpRepr)

        minimalVariableReplacement, newNodeRep = minimizeVariableReplacement(variableReplacement,
                                                                             templateNode.operatorRepresentation)
        for key, value in newNodeRep.items():
            templateNode.operatorRepresentation[key] = value

        ctxt, executionBlock, applicable = self.generateTilingLoop(ctxt, executionBlock, nodeMemoryConstraint,
                                                                   tilingSchedules, minimalVariableReplacement,
                                                                   operatorRepresentation)
        if applicable:
            ctxt, executionBlock = self.argStructGeneration.apply(ctxt, executionBlock, name)

        self._deinitPrefix()

        return ctxt, executionBlock
