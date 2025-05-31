# ----------------------------------------------------------------------
#
# File: PULPClusterTiling.py
#
# Last edited: 17.10.2023
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
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from Deeploy.AbstractDataTypes import VoidType
from Deeploy.DeeployTypes import CodeSnippet, ExecutionBlock, NetworkContext, NodeTemplate, OperatorRepresentation, \
    VariableBuffer, _ReferenceBuffer
from Deeploy.TilingExtension.CodeTransformationPasses.TilingCodeGeneration import TilingCodeGeneration, dictOfArrays
from Deeploy.TilingExtension.CodeTransformationPasses.TilingPrototypes import ProfilingSingleBufferingTilingMixIn, \
    SingleBufferingTilingMixIn, TilingMetaInfo
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint, TensorMemoryConstraint
from Deeploy.TilingExtension.TilingCodegen import HyperRectangle, TilingSchedule, VariableReplacementScheme, \
    calculateFlatOffset, minimizeRectangle, stridesFromShape

_openTileLoopTemplate = NodeTemplate("""
// TILING LOOP
for (int TILING_I=${numTiles}[*${tileIdxPtr}]; TILING_I<${numTiles}[(*${tileIdxPtr})+1]; TILING_I++){
""")

_closeTileLoopTemplate = NodeTemplate("""
// CLOSE TILING LOOP
}
*${tileIdxPtr} += 1;

""")

_transfer1DTemplate = NodeTemplate("""
// MOVE TILE ${innerTilePtr} from ${outerTilePtr}
mchan_transfer_1d(\\
% if isinstance(cmd, str):
${cmd}[${tileIdxVar}]\\
% else:
${cmd}\\
% endif
, ${loc}, ${ext});
""")

_transfer2DTemplate = NodeTemplate("""
// MOVE TILE ${innerTilePtr} from ${outerTilePtr}
mchan_transfer_2d_ext_strided(\\
% if isinstance(cmd, str):
${cmd}[${tileIdxVar}]\\
% else:
${cmd}\\
% endif
, ${loc}, ${ext}, \\
% if isinstance(size_1d, str):
${size_1d}[${tileIdxVar}], \\
% else:
${size_1d}, \\
% endif
% if isinstance(stride_2d, str):
${stride_2d}[${tileIdxVar}]);
% else:
${stride_2d});
% endif
""")

_transferMultiDimensionalTemplate = NodeTemplate("""
// MOVE TILE ${innerTilePtr} from ${outerTilePtr}
% for size in shape[:-2]:
% if isinstance(size, str):
for (uint32_t i_dim_${loop.index} = 0; i_dim_${loop.index} < ${size}[${tileIdxVar}]; i_dim_${loop.index}++) {
% else:
for (uint32_t i_dim_${loop.index} = 0; i_dim_${loop.index} < ${size}; i_dim_${loop.index}++) {
% endif
% endfor
    const uint32_t ext_offset = \\
% for stride in stridesExt[:-2]:
% if isinstance(stride, str):
i_dim_${loop.index} * ${stride}[${tileIdxVar}]${'+ ' if not loop.last else ''}\\
% else:
i_dim_${loop.index} * ${stride}${'+ ' if not loop.last else ''}\\
% endif
% endfor
;
    const uint32_t loc_offset = \\
% for stride in stridesLoc[:-2]:
% if isinstance(stride, str):
i_dim_${loop.index} * ${stride}[${tileIdxVar}]${'+ ' if not loop.last else ''}\\
% else:
i_dim_${loop.index} * ${stride}${'+ ' if not loop.last else ''}\\
% endif
% endfor
;
    mchan_transfer_2d_ext_strided(\\
% if isinstance(cmd, str):
${cmd}[${tileIdxVar}], \\
% else:
${cmd}, \\
% endif
(void *)${loc} + loc_offset, (void *)${ext} + ext_offset, \\
% if isinstance(shape[-1], str):
${shape[-1]}[${tileIdxVar}], \\
% else:
${shape[-1]}, \\
% endif
% if isinstance(stridesExt[-2], str):
${stridesExt[-2]}[${tileIdxVar}]);
% else:
${stridesExt[-2]});
% endif
% for _ in shape[:-2]:
}
% endfor
""")

_blockTransferTemplate = NodeTemplate("""
// BLOCKING UNTIL ALL TRANSFERS FINISH
mchan_channel_wait(${channel_id});
""")

_updateReferenceTemplate = NodeTemplate("""
// UPDATE VARIABLE ${reference}
*${reference} = ${baseReference}[${tileNum}];
""")

_relativeOffsetReferenceUpdateTemplate = NodeTemplate("""
// UPDATE VARIABLE ${reference}
% if isinstance(relativeOffset, str):
${reference} += ${relativeOffset}[${tileIdxVar}];
% else:
${reference} += ${relativeOffset};
% endif
""")

_initDmaTemplate = NodeTemplate("""
uint32_t ${channel_id} = mchan_get_channel_id();
""")

_releaseDmaTemplate = NodeTemplate("""
mchan_channel_free(${channel_id});
""")

# ADD NUM TRANSFERS VARIABLE


def padShape(shape: Tuple[int, ...], rank: int) -> Tuple[int, ...]:
    assert rank >= len(
        shape), f"Cannot pad to rank smaller then shape's. Received rank: {rank}, shape rank: {len(shape)}"
    ret = tuple([1] * (rank - len(shape))) + shape
    assert len(ret) == rank
    return ret


def padOffset(offset: Tuple[int, ...], rank: int) -> Tuple[int, ...]:
    assert rank >= len(
        offset), f"Cannot pad to rank smaller then offset's. Received rank: {rank}, offset rank: {len(offset)}"
    ret = tuple([0] * (rank - len(offset))) + offset
    assert len(ret) == rank
    return ret


class PULPClusterTilingSB(TilingCodeGeneration):

    _prefix = "TILING_REPLACED_"

    _openTileLoopTemplate = _openTileLoopTemplate
    _closeTileLoopTemplate = _closeTileLoopTemplate

    _transfer1DTemplate = _transfer1DTemplate
    _transfer2DTemplate = _transfer2DTemplate
    _transferMultiDimensionalTemplate = _transferMultiDimensionalTemplate
    _blockTransferTemplate = _blockTransferTemplate

    _updateReferenceTemplate = _updateReferenceTemplate
    _relativeOffsetReferenceUpdateTemplate = _relativeOffsetReferenceUpdateTemplate

    _initDmaTemplate = _initDmaTemplate
    _releaseDmaTemplate = _releaseDmaTemplate

    @property
    def prefix(self):
        return self._prefix + self.targetMemLevel + "_"

    @classmethod
    def _transferOpRepr(cls, rect: HyperRectangle, localBuffer: VariableBuffer, externalBuffer: VariableBuffer,
                        direction: Literal["To",
                                           "From"], commonOpRepr: OperatorRepresentation) -> OperatorRepresentation:
        assert len(rect.dims) == len(externalBuffer.shape), "Rectangle rank should be the same as external buffer's"

        operatorRepresentation = commonOpRepr.copy()
        operatorRepresentation.update({
            "loc": localBuffer.name,
            "ext": externalBuffer.name,
        })

        mchanFlags = 0
        mchanFlags += (1 << 0) if direction == "To" else 0  # direction
        mchanFlags += (1 << 1)  # increment addresses
        mchanFlags += (1 << 2) if len(rect.dims) >= 2 else 0  # 2d transfer
        mchanFlags += (1 << 3)  # event enable

        if len(rect.dims) == 1:
            mchanTransferSize = rect.dims[0]
        elif len(rect.dims) == 2:
            mchanTransferSize = int(np.prod(rect.dims))
        else:
            mchanTransferSize = int(np.prod(rect.dims[-2:]))

        assert mchanTransferSize <= 2**17, f"The Dma transfer size for mchan should be representable with 17 bits, current number of bits required is {np.ceil(np.log2(mchanTransferSize))}"

        operatorRepresentation["cmd"] = (mchanFlags << 17) + mchanTransferSize

        localBufferStrides = stridesFromShape(rect.dims)
        externalBufferStrides = stridesFromShape(externalBuffer.shape)

        if len(rect.dims) == 2:
            operatorRepresentation["size_1d"] = rect.dims[-1]
            operatorRepresentation["stride_2d"] = externalBufferStrides[-2]
        elif len(rect.dims) > 2:
            operatorRepresentation["shape"] = rect.dims
            operatorRepresentation["stridesExt"] = externalBufferStrides
            operatorRepresentation["stridesLoc"] = localBufferStrides

        return operatorRepresentation

    def _generateVariableUpdates(self, tilingSchedule: TilingSchedule, variableReplacement: VariableReplacementScheme,
                                 ctxt: NetworkContext,
                                 operatorRepresentation: OperatorRepresentation) -> List[CodeSnippet]:
        updates = []

        for key in variableReplacement.perTileReplacements.keys():
            buf = ctxt.lookup(operatorRepresentation[key])
            reference = str(buf._instance)

            updates.append(
                CodeSnippet(self._updateReferenceTemplate, {
                    "reference": reference,
                    "tileNum": "TILING_I",
                    "baseReference": buf._referenceName
                }))

        return updates

    def _legalizeTransfers(self, transfers: List[HyperRectangle], outerShape: Tuple[int, ...], typeWidth: int,
                           isFinalMemoryLevel: bool) -> Tuple[List[HyperRectangle], Tuple[int, ...]]:
        transfersCommonRank = max(len(rect.dims) for rect in transfers)
        commonRank = max(transfersCommonRank, len(outerShape))
        outerShape = padShape(outerShape, commonRank)

        minOuterShape = None

        if isFinalMemoryLevel:
            minTransfers = []
            for rect in transfers:
                paddedRect = HyperRectangle(padOffset(rect.offset, commonRank), padShape(rect.dims, commonRank))
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
                minTransfers.append(minRect)
        else:
            minTransfers = [HyperRectangle((0,), (int(np.prod(rect.dims)),)) for rect in transfers]
            minOuterShape = (int(np.prod(outerShape)),)

        if minOuterShape is not None:
            outerShape = minOuterShape
        transfers = minTransfers

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

    def _generateDmaTransferCall(self, ctxt: NetworkContext, nodeName: str, tensorName: str,
                                 transfers: List[HyperRectangle], tileIdxVar: str, localBuffer: VariableBuffer,
                                 externalBuffer: VariableBuffer, direction: Literal["To", "From"]) -> CodeSnippet:
        commonOpRepr = {
            "innerTilePtr": localBuffer.name,
            "outerTilePtr": externalBuffer.name,
            "tileIdxVar": tileIdxVar,
        }

        assert all(len(transfers[0].dims) == len(rect.dims) for rect in transfers), \
            "Currently supporting only rectangles of same rank"

        assert len(transfers[0].dims) == len(externalBuffer.shape), \
            "External buffer's rank should be equal to the internal buffer's"

        transferOpReprs = [
            self._transferOpRepr(rect, localBuffer, externalBuffer, direction, commonOpRepr) for rect in transfers
        ]

        transferOpRepr = self._hoistOpReprUpdates(ctxt, transferOpReprs, nodeName, f"{nodeName}_{tensorName}_")

        dimLen = len(transfers[0].dims)
        assert dimLen > 0
        if dimLen == 1:
            template = self._transfer1DTemplate
        elif dimLen == 2:
            template = self._transfer2DTemplate
        else:
            template = self._transferMultiDimensionalTemplate

        return CodeSnippet(template, transferOpRepr)

    def _generateExternalReferenceUpdate(self, ctxt: NetworkContext, nodeName: str, tensorName: str,
                                         transfers: List[HyperRectangle], tileIdxVar: str,
                                         externalBuffer: VariableBuffer) -> Optional[CodeSnippet]:
        externalBufferStrides = stridesFromShape(externalBuffer.shape)
        offsets = [calculateFlatOffset(rect.offset, externalBufferStrides) for rect in transfers]
        relativeOffsets = [_next - _prev for _prev, _next in zip(offsets[:-1], offsets[1:])]

        if len(relativeOffsets) == 0 or all(offset == 0 for offset in relativeOffsets):
            return None

        updateRefOpRepr: OperatorRepresentation = {"reference": externalBuffer.name, "tileIdxVar": tileIdxVar}

        if all(relativeOffsets[0] == offset for offset in relativeOffsets):
            updateRefOpRepr["relativeOffset"] = relativeOffsets[0]
        else:
            relativeOffsets.append(0)  # To have the same length as the number of tiles
            buffer = self._hoistValues(ctxt, f'{nodeName}_{tensorName}_relativeOffset', relativeOffsets, nodeName)
            updateRefOpRepr["relativeOffset"] = buffer.name
        return CodeSnippet(self._relativeOffsetReferenceUpdateTemplate, updateRefOpRepr)

    def _generateDmaTransferCallsAndExternalReferenceUpdates(
            self, ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation,
            transferSchedule: List[Dict[str, HyperRectangle]], tensorMemoryConstraintDict: Dict[str,
                                                                                                TensorMemoryConstraint],
            tileIdxVar: str, direction: Literal["To",
                                                "From"]) -> Tuple[NetworkContext, List[CodeSnippet], List[CodeSnippet]]:
        nodeName = operatorRepresentation["nodeName"]
        transferCalls: List[CodeSnippet] = []
        referenceUpdates: List[CodeSnippet] = []

        for tensorName, rects in dictOfArrays(transferSchedule).items():
            l1Buffer = ctxt.lookup(operatorRepresentation[tensorName])
            assert isinstance(l1Buffer, _ReferenceBuffer)
            l2Buffer = ctxt.lookup(l1Buffer._referenceName)
            assert isinstance(l2Buffer, VariableBuffer)
            tensorMemoryConstraint = tensorMemoryConstraintDict[l2Buffer.name]
            l2BufferShape = tensorMemoryConstraint.memoryConstraints['L2'].shape
            assert l2BufferShape is not None

            rects, l2BufferShape = self._legalizeTransfers(
                rects, tuple(l2BufferShape), l1Buffer._type.referencedType.typeWidth,
                self.isFinalMemoryLevel(tensorMemoryConstraint, l1Buffer._memoryLevel))

            l2BufferRef = ctxt.hoistReference(f"{nodeName}_{l2Buffer.name}_tiling_ref", l2Buffer, VoidType)
            l2BufferRef._memoryLevel = self.targetMemLevel
            l2BufferRef.shape = l2BufferShape

            transferCalls.append(
                self._generateDmaTransferCall(ctxt, nodeName, tensorName, rects, tileIdxVar, l1Buffer, l2BufferRef,
                                              direction))
            referenceUpdate = self._generateExternalReferenceUpdate(ctxt, nodeName, tensorName, rects, tileIdxVar,
                                                                    l2BufferRef)
            if referenceUpdate is not None:
                referenceUpdates.append(referenceUpdate)

        return ctxt, transferCalls, referenceUpdates

    def _tilingLoop(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
                    nodeMemoryConstraint: NodeMemoryConstraint, tilingSchedule: TilingSchedule,
                    variableReplacement: VariableReplacementScheme,
                    operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, ExecutionBlock, bool]:
        tileIdxPtr = self._hoistTileIdxPtr(ctxt, operatorRepresentation)

        ctxt, ingressDmaTransferCalls, ingressReferenceUpdates = self._generateDmaTransferCallsAndExternalReferenceUpdates(
            ctxt, operatorRepresentation, tilingSchedule.inputLoadSchedule,
            nodeMemoryConstraint.inputTensorMemoryConstraints, "TILING_I", "To")
        ctxt, egressDmaTransferCalls, egressReferenceUpdates = self._generateDmaTransferCallsAndExternalReferenceUpdates(
            ctxt, operatorRepresentation, tilingSchedule.outputLoadSchedule,
            nodeMemoryConstraint.outputTensorMemoryConstraints, "TILING_I", "From")

        dmaWaitCall = CodeSnippet(self._blockTransferTemplate, {"channel_id": "channel_id"})

        for cs in ingressReferenceUpdates + egressReferenceUpdates:
            assert "relativeOffset" in cs.operatorRepresentation

        openLoopStatement = [
            CodeSnippet(self._openTileLoopTemplate, {
                "numTiles": operatorRepresentation["numTiles"],
                "tileIdxPtr": tileIdxPtr
            })
        ]

        closeLoopStatement = [
            CodeSnippet(self._closeTileLoopTemplate, {
                "numTiles": operatorRepresentation["numTiles"],
                "tileIdxPtr": tileIdxPtr
            })
        ]

        setupStatements = [CodeSnippet(self._initDmaTemplate, {"channel_id": "channel_id"})]
        teardownStatements = [CodeSnippet(self._releaseDmaTemplate, {"channel_id": "channel_id"})]

        variableUpdates = self._generateVariableUpdates(tilingSchedule, variableReplacement, ctxt,
                                                        operatorRepresentation)

        metaInfo = TilingMetaInfo(nodeName = operatorRepresentation['nodeName'] + "_L2",
                                  nodeOps = operatorRepresentation['nodeOps'],
                                  numTiles = len(tilingSchedule.outputLoadSchedule),
                                  tileIdxVar = "TILING_I",
                                  kernelLevelTiling = True)

        newExecutionBlock = self.generateAllTilingCode(
            executionBlock, metaInfo, ingressDmaTransferCalls, [dmaWaitCall], [], egressDmaTransferCalls, [dmaWaitCall],
            [], variableUpdates, openLoopStatement,
            ingressReferenceUpdates + egressReferenceUpdates + closeLoopStatement, setupStatements, teardownStatements)

        return ctxt, newExecutionBlock, True

    def generateTilingLoop(
            self, ctxt: NetworkContext, executionBlock: ExecutionBlock, nodeMemoryConstraint: NodeMemoryConstraint,
            tilingSchedules: List[TilingSchedule], variableReplacement: VariableReplacementScheme,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, ExecutionBlock, bool]:

        flatTilingSchedule = copy.copy(tilingSchedules[0])
        for tilingSchedule in tilingSchedules[1:]:
            flatTilingSchedule += tilingSchedule

        # SCHEREMO: hoist numTiles

        offsetLists = list({**flatTilingSchedule.inputBaseOffsets, **flatTilingSchedule.outputBaseOffsets}.values())

        if len(offsetLists) == 0:
            return ctxt, executionBlock, False

        for offsetList in offsetLists:
            if not len(offsetList) == 1:
                return ctxt, executionBlock, False

        operatorRepresentation["numTiles"] = self._hoistNumTiles(ctxt, operatorRepresentation['nodeName'],
                                                                 tilingSchedules)

        return self._tilingLoop(ctxt, executionBlock, nodeMemoryConstraint, flatTilingSchedule, variableReplacement,
                                operatorRepresentation)


class PULPClusterTilingGenerationSB(PULPClusterTilingSB, SingleBufferingTilingMixIn):
    pass


class ProfilingPULPClusterTilingGenerationSB(PULPClusterTilingSB, ProfilingSingleBufferingTilingMixIn):
    pass
