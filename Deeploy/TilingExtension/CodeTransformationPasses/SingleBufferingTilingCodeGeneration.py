# ----------------------------------------------------------------------
#
# File: PULPL3TilingSB.py
#
# Last edited: 19.04.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
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
from typing import Dict, List, Set, Tuple

from Deeploy.AbstractDataTypes import VoidType
from Deeploy.DeeployTypes import CodeSnippet, ExecutionBlock, NetworkContext, OperatorRepresentation, VariableBuffer, \
    _ReferenceBuffer
from Deeploy.TilingExtension.AsyncDma import AsyncDma, DmaDirection, Future
from Deeploy.TilingExtension.CodeTransformationPasses.TilingCodeGeneration import TilingCodeGeneration
from Deeploy.TilingExtension.CodeTransformationPasses.TilingHoistingMixIn import dictOfArrays
from Deeploy.TilingExtension.CodeTransformationPasses.TilingPrototypes import TilingMetaInfo
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint, TensorMemoryConstraint
from Deeploy.TilingExtension.TilingCodegen import HyperRectangle, TilingSchedule, VariableReplacementScheme


class SingleBufferingTilingCodeGeneration(TilingCodeGeneration):

    def __init__(self, externalMemory: str, localMemory: str, dma: AsyncDma):
        super().__init__(externalMemory, localMemory, dma, 1)

    def _generateDmaTransferCallsAndExternalReferenceUpdates(
            self, ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation,
            transferSchedule: List[Dict[str, HyperRectangle]],
            tensorMemoryConstraintDict: Dict[str, TensorMemoryConstraint], tileIdxVar: str,
            direction: DmaDirection) -> Tuple[NetworkContext, List[CodeSnippet], List[CodeSnippet], Set[Future]]:
        transferCalls: List[CodeSnippet] = []
        referenceUpdates: List[CodeSnippet] = []
        futures: Set[Future] = set()

        for tensorName, rects in dictOfArrays(transferSchedule).items():
            localBuffer = ctxt.lookup(operatorRepresentation[tensorName])
            assert localBuffer._memoryLevel == self.localMemory
            assert isinstance(localBuffer, _ReferenceBuffer)
            externalBuffer = ctxt.lookup(localBuffer._referenceName)
            assert isinstance(externalBuffer, VariableBuffer)
            tensorMemoryConstraint = tensorMemoryConstraintDict[externalBuffer.name]
            externalBufferShape = tensorMemoryConstraint.memoryConstraints[self.externalMemory].shape
            assert externalBufferShape is not None

            rects, externalBufferShape = self._legalizeTransfers(rects, tuple(externalBufferShape),
                                                                 localBuffer._type.referencedType.typeWidth,
                                                                 self.isFinalMemoryLevel(tensorMemoryConstraint))

            externalBufferRef = self._hoistReference(ctxt,
                                                     externalBuffer.name + "_ref",
                                                     externalBuffer,
                                                     override_type = VoidType)
            externalBufferRef.shape = externalBufferShape

            future = self.dma.getFuture(tensorName, direction)
            futures.add(future)

            transferCalls.extend(
                self._generateDmaTransferCalls(ctxt, tensorName, rects, tileIdxVar, localBuffer, externalBufferRef,
                                               direction, future))

            referenceUpdate = self._generateExternalReferenceUpdate(ctxt, tensorName, rects, tileIdxVar,
                                                                    externalBufferRef)
            if referenceUpdate is not None:
                referenceUpdates.append(referenceUpdate)

        return ctxt, transferCalls, referenceUpdates, futures

    def _tilingLoop(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
                    nodeMemoryConstraint: NodeMemoryConstraint, tilingSchedule: TilingSchedule,
                    variableReplacement: VariableReplacementScheme,
                    operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, ExecutionBlock, bool]:
        ctxt, ingressDmaTransferCalls, ingressReferenceUpdates, ingressAsyncTokens = self._generateDmaTransferCallsAndExternalReferenceUpdates(
            ctxt, operatorRepresentation, tilingSchedule.inputLoadSchedule,
            nodeMemoryConstraint.inputTensorMemoryConstraints, "TILING_I", "ExternalToLocal")
        ctxt, egressDmaTransferCalls, egressReferenceUpdates, egressAsyncTokens = self._generateDmaTransferCallsAndExternalReferenceUpdates(
            ctxt, operatorRepresentation, tilingSchedule.outputLoadSchedule,
            nodeMemoryConstraint.outputTensorMemoryConstraints, "TILING_I", "LocalToExternal")

        ingressDmaWaitStatements = [token.wait() for token in ingressAsyncTokens]
        egressDmaWaitStatements = [token.wait() for token in egressAsyncTokens]

        setupStatements = self.dma.setup()
        setupStatements += [token.init() for token in ingressAsyncTokens | egressAsyncTokens]

        teardownStatements = self.dma.teardown()
        teardownStatements += [token.deinit() for token in ingressAsyncTokens | egressAsyncTokens]

        openLoopStatement = [CodeSnippet(self._openTileLoopTemplate, {**operatorRepresentation})]
        closeLoopStatement = [CodeSnippet(self._closeTileLoopTemplate, {**operatorRepresentation})]

        metaInfo = TilingMetaInfo(
            nodeName = operatorRepresentation['nodeName'] + f"_{self.externalMemory}",
            nodeOps = operatorRepresentation['nodeOps'],
            numTiles = len(tilingSchedule.outputLoadSchedule),
            tileIdxVar = "TILING_I",
            kernelLevelTiling = self.localMemory == 'L1')  # HACK: temporary hack until reworking profiling

        newExecutionBlock = self.generateAllTilingCode(
            executionBlock, metaInfo, ingressDmaTransferCalls, ingressDmaWaitStatements, [], egressDmaTransferCalls,
            egressDmaWaitStatements, [], [], openLoopStatement,
            ingressReferenceUpdates + egressReferenceUpdates + closeLoopStatement, setupStatements, teardownStatements)

        return ctxt, newExecutionBlock, True

    def generateTilingLoop(
            self, ctxt: NetworkContext, executionBlock: ExecutionBlock, nodeMemoryConstraint: NodeMemoryConstraint,
            tilingSchedules: List[TilingSchedule], variableReplacement: VariableReplacementScheme,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, ExecutionBlock, bool]:

        flatTilingSchedule = copy.copy(tilingSchedules[0])
        for tilingSchedule in tilingSchedules[1:]:
            flatTilingSchedule += tilingSchedule

        offsetLists = list({**flatTilingSchedule.inputBaseOffsets, **flatTilingSchedule.outputBaseOffsets}.values())

        if len(offsetLists) == 0:
            return ctxt, executionBlock, False

        for offsetList in offsetLists:
            if not len(offsetList) == self.bufferCount:
                return ctxt, executionBlock, False

        numTiles, tileIdxPtr = self._hoistTileNumAndIdxPtr(ctxt, tilingSchedules)
        operatorRepresentation["numTiles"] = numTiles.name
        operatorRepresentation["tileIdxPtr"] = tileIdxPtr.name

        return self._tilingLoop(ctxt, executionBlock, nodeMemoryConstraint, flatTilingSchedule, variableReplacement,
                                operatorRepresentation)
