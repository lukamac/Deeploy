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

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np

from Deeploy.AbstractDataTypes import PointerClass, VoidType
from Deeploy.DeeployTypes import CodeSnippet, ExecutionBlock, NetworkContext, NodeTemplate, OperatorRepresentation, \
    VariableBuffer, _ReferenceBuffer
from Deeploy.Targets.PULPOpen.CodeTransformationPasses.PULPClusterTilingSB import PULPClusterTilingSB
from Deeploy.TilingExtension.CodeTransformationPasses.TilingCodeGeneration import dictOfArrays
from Deeploy.TilingExtension.CodeTransformationPasses.TilingPrototypes import DoubleBufferingTilingMixIn, \
    ProfilingDoubleBufferingTilingMixIn, TilingMetaInfo
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint, TensorMemoryConstraint
from Deeploy.TilingExtension.TilingCodegen import TilingSchedule, VariableReplacementScheme

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


class OffsettedReferenceBuffer(_ReferenceBuffer):
    allocTemplate = NodeTemplate("${type.typeName} ${name} = (${type.typeName})${referenceName} + ${offset};")

    def __init__(self, name: str = '', shape = [1], reference: Optional[VariableBuffer] = None, offset: int = 0):
        super().__init__(name, shape, reference)
        self._offset = offset

    def _bufferRepresentation(self) -> Dict:
        repr = super()._bufferRepresentation()
        repr['offset'] = self._offset
        return repr


class PULPClusterTilingDB(PULPClusterTilingSB):

    _chooseBufferTemplate = _chooseBufferTemplate

    def _hoistMultibufferReferences(self, ctxt: NetworkContext, referenceBuffer: VariableBuffer,
                                    tensorMemoryConstraint: TensorMemoryConstraint) -> List[_ReferenceBuffer]:
        memoryConstraint = tensorMemoryConstraint.memoryConstraints[self.targetMemLevel]
        assert memoryConstraint.addrSpace is not None, "Assuming address space is set"
        totalSize = memoryConstraint.addrSpace[1] - memoryConstraint.addrSpace[0]
        assert isinstance(memoryConstraint.multiBufferCoefficient,
                          int), "Assuming multi buffer coefficient has been assigned"
        assert totalSize % memoryConstraint.multiBufferCoefficient == 0, "Assuming total size is divisible by the multi buffer coefficient"
        bufferSize = totalSize // memoryConstraint.multiBufferCoefficient

        assert memoryConstraint.multiBufferCoefficient == 2, "Multi buffer coefficient has to be equal to 2 since this is for double buffering"
        assert memoryConstraint.shape is not None
        assert len(memoryConstraint.shape) > 0
        assert isinstance(memoryConstraint.shape[0], int)
        tileLength = np.prod(memoryConstraint.shape)
        tileSize = int(np.ceil(tileLength * referenceBuffer._type.referencedType.typeWidth / 8))

        assert bufferSize >= tileSize, f"Provided buffer size is not enough to fit the tile. Buffer size: {bufferSize}, tile size: {tileSize}"

        multibufferReferences = []
        for i in range(memoryConstraint.multiBufferCoefficient):
            buffer = OffsettedReferenceBuffer(
                name = f"{referenceBuffer.name}_buffer_{i}",
                shape = memoryConstraint.shape,
                reference = referenceBuffer,
                offset = i * bufferSize,
            )
            buffer._type = PointerClass(VoidType)
            buffer._memoryLevel = self.targetMemLevel
            ctxt.add(buffer)
            buffer._instance = buffer._type(buffer.name, ctxt = ctxt)
            multibufferReferences.append(buffer)

        return multibufferReferences

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

        tileIdxPtr = self._hoistTileIdxPtr(ctxt, operatorRepresentation)
        nodeName = operatorRepresentation['nodeName']

        setupStatements: List[CodeSnippet] = [CodeSnippet(self._initDmaTemplate, {"channel_id": "channel_id"})]

        openLoopStatements: List[CodeSnippet] = [
            CodeSnippet(self._openTileLoopTemplate, {
                "numTiles": operatorRepresentation["numTiles"],
                "tileIdxPtr": tileIdxPtr
            })
        ]

        ingressLoopDmaTransferCalls: List[CodeSnippet] = [
            CodeSnippet(_moveTileInCheckOpenStatement, {
                "numTiles": operatorRepresentation["numTiles"],
                "tileIdxVar": "TILING_I+1",
                "tileIdxPtr": tileIdxPtr
            })
        ]

        for tensorName, rectangles in dictOfArrays(tilingSchedule.inputLoadSchedule).items():
            l1Buffer = ctxt.lookup(operatorRepresentation[tensorName])
            assert isinstance(l1Buffer, _ReferenceBuffer)
            l2Buffer = ctxt.lookup(l1Buffer._referenceName)
            assert isinstance(l2Buffer, VariableBuffer)
            tensorMemoryConstraint = nodeMemoryConstraint.inputTensorMemoryConstraints[l2Buffer.name]
            l2BufferShape = tensorMemoryConstraint.memoryConstraints['L2'].shape
            assert l2BufferShape is not None

            rectangles, l2BufferShape = self._legalizeTransfers(
                rectangles, tuple(l2BufferShape), l1Buffer._type.referencedType.typeWidth,
                self.isFinalMemoryLevel(tensorMemoryConstraint, l1Buffer._memoryLevel))

            l2BufferRef = ctxt.hoistReference(f"{nodeName}_{l2Buffer.name}_tiling_ref", l2Buffer, VoidType)
            l2BufferRef._memoryLevel = self.targetMemLevel
            l2BufferRef.shape = l2BufferShape

            tensorMemoryConstraint = nodeMemoryConstraint.inputTensorMemoryConstraints[l2Buffer.name]
            l1BuffersReferences = self._hoistMultibufferReferences(ctxt, l1Buffer, tensorMemoryConstraint)

            nextLocalBufferReference = _ReferenceBuffer(f"{l1Buffer.name}_next", reference = l1BuffersReferences[1])
            nextLocalBufferReference._type = l1BuffersReferences[1]._type
            nextLocalBufferReference._memoryLevel = self.targetMemLevel
            ctxt.add(nextLocalBufferReference, 'local')
            nextLocalBufferReference._instance = nextLocalBufferReference._type(f"{l1Buffer.name}_next", ctxt = ctxt)

            openLoopStatements.append(self._generateBufferChoice(l1BuffersReferences, l1Buffer, "TILING_I"))

            dmaTransferCall = self._generateDmaTransferCall(ctxt, nodeName, tensorName, rectangles, "TILING_I+1",
                                                            nextLocalBufferReference, l2BufferRef, 'To')

            ingressLoopDmaTransferCalls.append(
                self._generateBufferChoice(l1BuffersReferences, nextLocalBufferReference, "TILING_I+1"))
            ingressLoopDmaTransferCalls.append(dmaTransferCall)

            initialDmaTransferCall = CodeSnippet(dmaTransferCall.template,
                                                 operatorRepresentation = {
                                                     **dmaTransferCall.operatorRepresentation,
                                                     "tileIdxVar": 0,
                                                     "loc": l1Buffer.name,
                                                     "innerTilePtr": l1Buffer.name,
                                                 })
            setupStatements.append(initialDmaTransferCall)

            referenceUpdate = self._generateExternalReferenceUpdate(ctxt, nodeName, tensorName, rectangles,
                                                                    "TILING_I+1", l2BufferRef)
            if referenceUpdate is not None:
                ingressLoopDmaTransferCalls.append(referenceUpdate)
                initialReferenceUpdate = CodeSnippet(referenceUpdate.template,
                                                     operatorRepresentation = {
                                                         **referenceUpdate.operatorRepresentation,
                                                         "tileIdxVar": 0,
                                                     })
                setupStatements.append(initialReferenceUpdate)

        ingressLoopDmaTransferCalls.append(CodeSnippet(_moveTileInCheckCloseStatement, {}))

        egressLoopDmaTransferCalls: List[CodeSnippet] = []

        for tensorName, rectangles in dictOfArrays(tilingSchedule.outputLoadSchedule).items():
            l1Buffer = ctxt.lookup(operatorRepresentation[tensorName])
            assert isinstance(l1Buffer, _ReferenceBuffer)
            l2Buffer = ctxt.lookup(l1Buffer._referenceName)
            assert isinstance(l2Buffer, VariableBuffer)
            tensorMemoryConstraint = nodeMemoryConstraint.outputTensorMemoryConstraints[l2Buffer.name]
            l2BufferShape = tensorMemoryConstraint.memoryConstraints['L2'].shape
            assert l2BufferShape is not None

            rectangles, l2BufferShape = self._legalizeTransfers(
                rectangles, tuple(l2BufferShape), l1Buffer._type.referencedType.typeWidth,
                self.isFinalMemoryLevel(tensorMemoryConstraint, l1Buffer._memoryLevel))

            l2BufferRef = ctxt.hoistReference(f"{nodeName}_{l2Buffer.name}_tiling_ref", l2Buffer, VoidType)
            l2BufferRef._memoryLevel = self.targetMemLevel
            l2BufferRef.shape = l2BufferShape

            tensorMemoryConstraint = nodeMemoryConstraint.outputTensorMemoryConstraints[l2Buffer.name]
            l1BuffersReferences = self._hoistMultibufferReferences(ctxt, l1Buffer, tensorMemoryConstraint)

            openLoopStatements.append(self._generateBufferChoice(l1BuffersReferences, l1Buffer, "TILING_I"))

            dmaTransferCall = self._generateDmaTransferCall(ctxt, nodeName, tensorName, rectangles, "TILING_I",
                                                            l1Buffer, l2BufferRef, 'From')
            egressLoopDmaTransferCalls.append(dmaTransferCall)

            referenceUpdate = self._generateExternalReferenceUpdate(ctxt, nodeName, tensorName, rectangles, "TILING_I",
                                                                    l2BufferRef)
            if referenceUpdate is not None:
                egressLoopDmaTransferCalls.append(referenceUpdate)

        dmaWaitCall = CodeSnippet(self._blockTransferTemplate, {"channel_id": "channel_id"})

        teardownStatements = [dmaWaitCall, CodeSnippet(self._releaseDmaTemplate, {"channel_id": "channel_id"})]

        variableUpdates = self._generateVariableUpdates(tilingSchedule, variableReplacement, ctxt,
                                                        operatorRepresentation)

        for transaction in variableUpdates:
            _operatorRepresentation = transaction.operatorRepresentation
            _operatorRepresentation["tileNum"] = "TILING_I"

        closeLoopStatements = [
            CodeSnippet(self._closeTileLoopTemplate, {
                "numTiles": operatorRepresentation["numTiles"],
                "tileIdxPtr": tileIdxPtr
            })
        ]

        metaInfo = TilingMetaInfo(nodeName = operatorRepresentation['nodeName'] + "_L2",
                                  nodeOps = operatorRepresentation['nodeOps'],
                                  numTiles = len(tilingSchedule.outputLoadSchedule),
                                  tileIdxVar = "TILING_I",
                                  kernelLevelTiling = True)

        executionBlock = self.generateAllTilingCode(executionBlock, metaInfo, ingressLoopDmaTransferCalls,
                                                    [dmaWaitCall], [], egressLoopDmaTransferCalls, [dmaWaitCall], [],
                                                    variableUpdates, openLoopStatements, closeLoopStatements,
                                                    setupStatements, teardownStatements)

        return ctxt, executionBlock, True

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
            if not len(offsetList) == 2:
                return ctxt, executionBlock, False

        allNumTiles = [len(schedule.outputLoadSchedule) for schedule in tilingSchedules]
        operatorRepresentation["numTiles"] = self._hoistNumTiles(ctxt, operatorRepresentation['nodeName'],
                                                                 tilingSchedules)

        return self._tilingLoop(ctxt, executionBlock, nodeMemoryConstraint, flatTilingSchedule, variableReplacement,
                                operatorRepresentation)


class PULPClusterTilingGenerationDB(PULPClusterTilingDB, DoubleBufferingTilingMixIn):
    pass


class ProfilingPULPClusterTilingGenerationDB(PULPClusterTilingDB, ProfilingDoubleBufferingTilingMixIn):
    pass
