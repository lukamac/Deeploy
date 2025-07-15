# ----------------------------------------------------------------------
#
# File: PULPClusterTilingDB.py
#
# Last edited: 25.10.2023
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

import math
from typing import List, Set, Tuple

from Deeploy.AbstractDataTypes import VoidType
from Deeploy.DeeployTypes import CodeSnippet, ExecutionBlock, NetworkContext, NodeTemplate, OperatorRepresentation, \
    VariableBuffer, _ReferenceBuffer
from Deeploy.TilingExtension.AsyncDma import AsyncDma, Future, MultidimDmaSnippetGenerator
from Deeploy.TilingExtension.CodeTransformationPasses.SingleBufferingTilingCodeGeneration import \
    SingleBufferingTilingCodeGeneration
from Deeploy.TilingExtension.CodeTransformationPasses.TilingHoistingMixIn import dictOfArrays
from Deeploy.TilingExtension.CodeTransformationPasses.TilingPrototypes import TilingMetaInfo
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TilingCodegen import TilingSchedule, VariableReplacementScheme, stridesFromShape


class DoubleBufferingTilingCodeGeneration(SingleBufferingTilingCodeGeneration):

    _moveTileInCheckOpenStatement = NodeTemplate("""
    // DOUBLE BUFFERING CHECK TILE LOAD
    if ((${tileIdxVar}) < ${numTiles}[*${tileIdxPtr}+1]) {
    """)

    _moveTileInCheckCloseStatement = NodeTemplate("""
    }
    """)

    _chooseBufferTemplate = NodeTemplate("""
    switch ((${tileIdxVar}) % ${len(bufferReferences)}) {
    % for ref in bufferReferences:
        case ${loop.index}: ${bufferChoiceReference} = (${bufferChoiceReferenceType})${ref}; break;
    % endfor
    }
    """)

    def __init__(self, externalMemory: str, localMemory: str, dma: AsyncDma):
        super().__init__(externalMemory, localMemory, dma)
        self.bufferCount = 2

    def _generateBufferChoice(self, buffersReferences: List[_ReferenceBuffer], bufferChoiceReference: VariableBuffer,
                              tileIdxVar: str) -> CodeSnippet:
        return CodeSnippet(template = self._chooseBufferTemplate,
                           operatorRepresentation = {
                               "tileIdxVar": tileIdxVar,
                               "bufferReferences": [buff.name for buff in buffersReferences],
                               "bufferChoiceReference": bufferChoiceReference.name,
                               "bufferChoiceReferenceType": bufferChoiceReference._type.typeName
                           })

    def _tilingLoop(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
                    nodeMemoryConstraint: NodeMemoryConstraint, tilingSchedule: TilingSchedule,
                    variableReplacement: VariableReplacementScheme,
                    operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, ExecutionBlock, bool]:

        setupStatements: List[CodeSnippet] = []
        teardownStatements: List[CodeSnippet] = []

        openLoopStatements: List[CodeSnippet] = [CodeSnippet(self._openTileLoopTemplate, {**operatorRepresentation})]

        ingressLoopDmaTransferCalls: List[CodeSnippet] = [
            CodeSnippet(self._moveTileInCheckOpenStatement, {
                **operatorRepresentation, "tileIdxVar": "TILING_I+1"
            })
        ]

        ingressFutures: Set[Future] = set()

        for tensorName, rectangles in dictOfArrays(tilingSchedule.inputLoadSchedule).items():
            localBuffer = ctxt.lookup(operatorRepresentation[tensorName])
            assert localBuffer._memoryLevel == self.localMemory
            assert isinstance(localBuffer, _ReferenceBuffer)
            externalBuffer = ctxt.lookup(localBuffer._referenceName)
            assert isinstance(externalBuffer, VariableBuffer)
            tensorMemoryConstraint = nodeMemoryConstraint.inputTensorMemoryConstraints[externalBuffer.name]
            externalBufferShape = tensorMemoryConstraint.memoryConstraints[self.externalMemory].shape
            assert externalBufferShape is not None

            rectangles, externalBufferShape = self._legalizeTransfers(rectangles, tuple(externalBufferShape),
                                                                      localBuffer._type.referencedType.typeWidth,
                                                                      self.isFinalMemoryLevel(tensorMemoryConstraint))

            externalBufferRef = self._hoistReference(ctxt,
                                                     externalBuffer.name + "_ref",
                                                     externalBuffer,
                                                     override_type = VoidType)
            externalBufferRef.shape = externalBufferShape

            tensorMemoryConstraint = nodeMemoryConstraint.inputTensorMemoryConstraints[externalBuffer.name]
            l1BuffersReferences = self._hoistMultibufferReferences(ctxt, localBuffer, tensorMemoryConstraint)

            nextLocalBufferReference = self._hoistReference(ctxt, f"{tensorName}_next", l1BuffersReferences[1])

            openLoopStatements.append(self._generateBufferChoice(l1BuffersReferences, localBuffer, "TILING_I"))

            future = self.dma.getFuture(tensorName, "ExternalToLocal")
            ingressFutures.add(future)

            ingressLoopDmaTransferCalls.append(
                self._generateBufferChoice(l1BuffersReferences, nextLocalBufferReference, "TILING_I+1"))
            ingressLoopDmaTransferCalls.extend(
                self._generateDmaTransferCalls(ctxt, tensorName, rectangles, "TILING_I+1", nextLocalBufferReference,
                                               externalBufferRef, "ExternalToLocal", future))

            gen = MultidimDmaSnippetGenerator(self.dma)

            initialFuture = self.dma.getFuture(tensorName + "_init", "ExternalToLocal")
            initialDmaTransferCalls = gen.transfer(ctxt, externalBufferRef, localBuffer, rectangles[0].dims,
                                                   stridesFromShape(externalBufferShape),
                                                   stridesFromShape(rectangles[0].dims), "ExternalToLocal",
                                                   initialFuture, math.prod(externalBufferShape))
            setupStatements.extend(initialDmaTransferCalls)

            referenceUpdate = self._generateExternalReferenceUpdate(ctxt, tensorName, rectangles, "TILING_I+1",
                                                                    externalBufferRef)
            if referenceUpdate is not None:
                ingressLoopDmaTransferCalls.append(referenceUpdate)
                initialReferenceUpdate = CodeSnippet(referenceUpdate.template,
                                                     operatorRepresentation = {
                                                         **referenceUpdate.operatorRepresentation,
                                                         "tileIdxVar": 0,
                                                     })
                setupStatements.append(initialReferenceUpdate)

        ingressLoopDmaTransferCalls.append(CodeSnippet(self._moveTileInCheckCloseStatement, {}))
        ingressLoopDmaWaitStatements = [f.wait() for f in ingressFutures]

        egressLoopDmaTransferCalls: List[CodeSnippet] = []
        egressFutures: Set[Future] = set()

        for tensorName, rectangles in dictOfArrays(tilingSchedule.outputLoadSchedule).items():
            localBuffer = ctxt.lookup(operatorRepresentation[tensorName])
            assert localBuffer._memoryLevel == self.localMemory
            assert isinstance(localBuffer, _ReferenceBuffer)
            externalBuffer = ctxt.lookup(localBuffer._referenceName)
            assert isinstance(externalBuffer, VariableBuffer)
            tensorMemoryConstraint = nodeMemoryConstraint.outputTensorMemoryConstraints[externalBuffer.name]
            externalBufferShape = tensorMemoryConstraint.memoryConstraints[self.externalMemory].shape
            assert externalBufferShape is not None

            rectangles, externalBufferShape = self._legalizeTransfers(rectangles, tuple(externalBufferShape),
                                                                      localBuffer._type.referencedType.typeWidth,
                                                                      self.isFinalMemoryLevel(tensorMemoryConstraint))

            externalBufferRef = self._hoistReference(ctxt,
                                                     externalBuffer.name + "_ref",
                                                     externalBuffer,
                                                     override_type = VoidType)
            externalBufferRef.shape = externalBufferShape

            tensorMemoryConstraint = nodeMemoryConstraint.outputTensorMemoryConstraints[externalBuffer.name]
            l1BuffersReferences = self._hoistMultibufferReferences(ctxt, localBuffer, tensorMemoryConstraint)

            openLoopStatements.append(self._generateBufferChoice(l1BuffersReferences, localBuffer, "TILING_I"))

            future = self.dma.getFuture(tensorName, "LocalToExternal")
            egressFutures.add(future)

            dmaTransferCalls = self._generateDmaTransferCalls(ctxt, tensorName, rectangles, "TILING_I", localBuffer,
                                                              externalBufferRef, "LocalToExternal", future)
            egressLoopDmaTransferCalls.extend(dmaTransferCalls)

            referenceUpdate = self._generateExternalReferenceUpdate(ctxt, tensorName, rectangles, "TILING_I",
                                                                    externalBufferRef)
            if referenceUpdate is not None:
                egressLoopDmaTransferCalls.append(referenceUpdate)

        egressLoopDmaWaitStatements = [f.wait() for f in egressFutures]

        teardownStatements.extend([f.wait() for f in egressFutures])

        setupStatements = [f.init() for f in ingressFutures | egressFutures] + setupStatements
        teardownStatements.extend(f.deinit() for f in ingressFutures | egressFutures)

        closeLoopStatements = [CodeSnippet(self._closeTileLoopTemplate, {**operatorRepresentation})]

        metaInfo = TilingMetaInfo(
            nodeName = operatorRepresentation['nodeName'] + f"_{self.externalMemory}",
            nodeOps = operatorRepresentation['nodeOps'],
            numTiles = len(tilingSchedule.outputLoadSchedule),
            tileIdxVar = "TILING_I",
            kernelLevelTiling = self.localMemory == "L1")  # HACK: temporary hack until we fix the profiling

        executionBlock = self.generateAllTilingCode(executionBlock, metaInfo, ingressLoopDmaTransferCalls,
                                                    ingressLoopDmaWaitStatements, [], egressLoopDmaTransferCalls,
                                                    egressLoopDmaWaitStatements, [], [], openLoopStatements,
                                                    closeLoopStatements, setupStatements, teardownStatements)

        return ctxt, executionBlock, True
