# ----------------------------------------------------------------------
#
# File: SnitchClusterTilingSB.py
#
# Last edited: 03.06.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
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

import copy
from collections import namedtuple
from typing import Dict, List, Literal, Tuple

from Deeploy.DeeployTypes import CodeSnippet, ExecutionBlock, NetworkContext, NodeTemplate, OperatorRepresentation
from Deeploy.Targets.Snitch.DataTypes import Snitch_DMA_copy
from Deeploy.TilingExtension.AsyncDma import AsyncDma
from Deeploy.TilingExtension.CodeTransformationPasses.TilingCodeGeneration import TilingCodeGeneration
from Deeploy.TilingExtension.CodeTransformationPasses.TilingPrototypes import SingleBufferingTilingMixIn, TilingMetaInfo
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TilingCodegen import HyperRectangle, TilingSchedule, VariableReplacementScheme, \
    calculateRectangleOffset, minimizeRectangleDims

_openTileLoopTemplate = NodeTemplate("""

// TILING LOOP
for (int TILING_I=${numTiles}[*${tileIdxPtr}]; TILING_I<${numTiles}[(*${tileIdxPtr})+1]; TILING_I++){
""")

_closeTileLoopTemplate = NodeTemplate("""

// CLOSE TILING LOOP
}
*${tileIdxPtr} += 1;
                                      
""")

_moveTileInTemplate = NodeTemplate("""

// IMPORT TILE ${innerTilePtr} from ${outerTilePtr}
if(snrt_is_dm_core()){ 
    ${stateReference}.tid = snrt_dma_start_2d(${stateReference}.dst,
                    ${stateReference}.src,
                    ${stateReference}.size,
                    ${stateReference}.dst_stride,
                    ${stateReference}.src_stride,
                    ${stateReference}.repeat);
}                                   
""")

_iteratedMoveTileInTemplate = NodeTemplate("""

""")

_blockTileInTemplate = NodeTemplate("""

// BLOCKING IMPORT TILE ${innerTilePtr}
if(snrt_is_dm_core()){                            
    // snrt_dma_wait(${stateReference}.tid);
    snrt_dma_wait_all();
}
""")

_moveTileOutTemplate = NodeTemplate("""

// EXPORT TILE ${innerTilePtr} to ${outerTilePtr}
if(snrt_is_dm_core()){ 
    ${stateReference}.tid = snrt_dma_start_2d(${stateReference}.dst,
                    ${stateReference}.src,
                    ${stateReference}.size,
                    ${stateReference}.dst_stride,
                    ${stateReference}.src_stride,
                    ${stateReference}.repeat);
}
""")

_blockTileOutTemplate = NodeTemplate("""

// BLOCKING EXPORT TILE ${innerTilePtr}
if(snrt_is_dm_core()){ 
    //snrt_dma_wait(${stateReference}.tid);
    snrt_dma_wait_all();
}
""")

_updateDMATransferStructTemplate = NodeTemplate("""
                                                
// UPDATE DMA STRUCT ${stateReference}
${stateReference}.dst = ((char*)${dstPtr}) + ${dstOffsetPtr}[${tileNum}];
${stateReference}.src = ((char*)${srcPtr}) + ${srcOffsetPtr}[${tileNum}];
${stateReference}.size = ${sizePtr}[${tileNum}];
${stateReference}.dst_stride = ${dstStridePtr}[${tileNum}];
${stateReference}.src_stride = ${srcStridePtr}[${tileNum}];
${stateReference}.repeat = ${repeatPtr}[${tileNum}];
""")

_updateReferenceTemplate = NodeTemplate("""

// UPDATE VARIABLE ${reference}
*${reference} = ${baseReference}[${tileNum}];
""")

_DMAUpdate = namedtuple("_DMAUpdate", "dst src size dst_stride src_stride repeat tid direction")


class SnitchClusterTilingSB(TilingCodeGeneration):

    _openTileLoopTemplate = _openTileLoopTemplate
    _closeTileLoopTemplate = _closeTileLoopTemplate

    _moveTileInTemplate = _moveTileInTemplate
    _iteratedMoveTileInTemplate = _iteratedMoveTileInTemplate
    _blockTileInTemplate = _blockTileInTemplate

    _moveTileOutTemplate = _moveTileOutTemplate
    _blockTileOutTemplate = _blockTileOutTemplate

    _updateDMATransferStructTemplate = _updateDMATransferStructTemplate
    _updateReferenceTemplate = _updateReferenceTemplate

    def __init__(self, externalMemory: str, localMemory: str, dma: AsyncDma):
        super().__init__(externalMemory, localMemory, dma, 1)

    def _DMAStructName(self, tensorName: str) -> str:
        return f"{self.prefix}{tensorName}_DMA"

    def _generatePointerUpdates(self, ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation,
                                loadSchedule: List[Dict[str,
                                                        HyperRectangle]], nodeMemoryConstraint: NodeMemoryConstraint,
                                tilingSchedule: TilingSchedule) -> Dict[str, _DMAUpdate]:
        updateDict = {}
        deltaOffsets = {}

        for idx, loadStep in enumerate(loadSchedule):
            for _, (key, rect) in enumerate(loadStep.items()):

                if key in tilingSchedule.outputBaseOffsets.keys():
                    baseOffsets = tilingSchedule.outputBaseOffsets[key]
                    direction = "FromL1"
                else:
                    baseOffsets = tilingSchedule.inputBaseOffsets[key]
                    direction = "ToL1"

                if key not in updateDict.keys():
                    updateDict[key] = []
                if key not in deltaOffsets.keys():
                    deltaOffsets[key] = 0

                referenceBuffer = ctxt.lookup(ctxt.lookup(operatorRepresentation[key])._referenceName)
                l1Buffer = ctxt.lookup(operatorRepresentation[key])
                assert l1Buffer._memoryLevel == self.localMemory

                tensorMemoryConstraint = nodeMemoryConstraint.tensorMemoryConstraints[l1Buffer._referenceName]
                finalMemoryLevel = self.isFinalMemoryLevel(tensorMemoryConstraint)

                struct = self._rectToDMAStruct(ctxt, rect, direction, l1Buffer.name, l1Buffer._referenceName,
                                               finalMemoryLevel)
                accOffset = calculateRectangleOffset(rect, referenceBuffer)

                lIdx = idx % len(baseOffsets)

                if direction == "ToL1":
                    src = accOffset
                    dst = baseOffsets[lIdx]
                else:
                    src = baseOffsets[lIdx]
                    dst = accOffset

                size = struct.value['size'].value
                dst_stride = struct.value['dst_stride'].value
                src_stride = struct.value['src_stride'].value
                repeat = struct.value['repeat'].value
                tid = struct.value['tid'].value

                sol = _DMAUpdate(dst, src, size, dst_stride, src_stride, repeat, tid, direction)

                deltaOffsets[key] = accOffset
                updateDict[key].append(sol)

        return updateDict

    @classmethod
    def _rectToDMAStruct(cls, ctxt: NetworkContext, rectangle: HyperRectangle, direction: Literal["ToL1", "FromL1"],
                         L1Name: str, L2Name: str, finalMemoryLevel: bool) -> Snitch_DMA_copy:

        referenceBuffer = ctxt.lookup(L2Name)

        rect, referenceRect = minimizeRectangleDims(rectangle, referenceBuffer)
        assert len(rect.dims) <= 3, "Snitch's iDMA only 2D transfers are supported!"

        if direction == "FromL1":
            _src = L1Name
            _dst = referenceBuffer.name
        else:
            _src = referenceBuffer.name
            _dst = L1Name

        transfer_size = rect.dims[-1] * (referenceBuffer._type.referencedType.typeWidth // 8)

        src_stride = 0
        dst_stride = 0
        repeat = 1
        if len(rect.dims) > 1:
            repeat = rect.dims[-2]
            if direction == "ToL1":
                dst_stride = rect.dims[-1] * (referenceBuffer._type.referencedType.typeWidth // 8)
                src_stride = referenceRect.dims[-1] * (referenceBuffer._type.referencedType.typeWidth // 8)
            else:
                dst_stride = referenceRect.dims[-1] * (referenceBuffer._type.referencedType.typeWidth // 8)
                src_stride = rect.dims[-1] * (referenceBuffer._type.referencedType.typeWidth // 8)

        struct = Snitch_DMA_copy(
            {
                "dst": _dst,
                "src": _src,
                "size": transfer_size,
                "dst_stride": dst_stride,
                "src_stride": src_stride,
                "repeat": repeat,
                "tid": 0
            }, ctxt)

        return struct

    def _hoistDMAUpdates(self, ctxt: NetworkContext, tensorName: str, updateList: List[_DMAUpdate],
                         operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict]:

        operatorRepresentation = operatorRepresentation.copy()

        dstList = []
        srcList = []
        sizeList = []
        dstStrideList = []
        srcStrideList = []
        repeatList = []
        for update in updateList:
            dstList.append(int(update.dst))
            srcList.append(int(update.src))
            sizeList.append(int(update.size))
            dstStrideList.append(int(update.dst_stride))
            srcStrideList.append(int(update.src_stride))
            repeatList.append(int(update.repeat))

        dmaName = self._DMAStructName(tensorName)

        operatorRepresentation['stateReference'] = dmaName
        operatorRepresentation['tileNum'] = "TILING_I"

        if updateList[0].direction == "ToL1":
            operatorRepresentation['dstPtr'] = ctxt.lookup(operatorRepresentation[tensorName]).name
            operatorRepresentation['srcPtr'] = ctxt.lookup(operatorRepresentation[tensorName])._referenceName

            dstOffsetList = [0] * len(updateList)
            srcOffsetList = [srcList[i] - srcList[0] for i in range(0, len(srcList))]
            # srcOffsetList = [0] + [sum(sizeList[:i+1]) for i in range(0, len(sizeList)-1)]
        else:
            operatorRepresentation['dstPtr'] = ctxt.lookup(operatorRepresentation[tensorName])._referenceName
            operatorRepresentation['srcPtr'] = ctxt.lookup(operatorRepresentation[tensorName]).name

            dstOffsetList = [dstList[i] - dstList[0] for i in range(0, len(dstList))]
            # dstOffsetList = [0] + [sum(sizeList[:i+1]) for i in range(0, len(sizeList)-1)]
            srcOffsetList = [0] * len(updateList)

        tensorUpdatesMap = {
            "dst_offset": (dstOffsetList, None),
            "dst_stride": (dstStrideList, Snitch_DMA_copy.structTypeDict['dst_stride']),
            "src_offset": (srcOffsetList, None),
            "src_stride": (srcStrideList, Snitch_DMA_copy.structTypeDict['src_stride']),
            "size": (sizeList, Snitch_DMA_copy.structTypeDict['size']),
            "repeat": (repeatList, Snitch_DMA_copy.structTypeDict['repeat']),
        }

        def reprPtrName(snake_case_name: str) -> str:
            substrings = snake_case_name.split("_")
            lowerCamelCaseName = "".join([substrings[0]] + [x.capitalize() for x in substrings[1:]])
            return lowerCamelCaseName + "Ptr"

        for name, (values, override_type) in tensorUpdatesMap.items():
            hoistName = f"{tensorName}_{name}"
            cb = self._hoistValues(ctxt, hoistName, values, override_type)
            ref = self._hoistReference(ctxt, hoistName + "_ref", cb)
            operatorRepresentation[reprPtrName(name)] = ref.name

        return ctxt, operatorRepresentation

    def _generateEgressPointerUpdates(
            self, nodeMemoryConstraint: NodeMemoryConstraint, tilingSchedule: TilingSchedule, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, List[CodeSnippet]]:

        updates = []
        newCtxt = ctxt.copy()

        updateDict = self._generatePointerUpdates(ctxt, operatorRepresentation, tilingSchedule.outputLoadSchedule,
                                                  nodeMemoryConstraint, tilingSchedule)

        for key, updateList in updateDict.items():

            newCtxt, newNodeRep = self._hoistDMAUpdates(newCtxt, key, updateList, operatorRepresentation)
            updates.append(CodeSnippet(self._updateDMATransferStructTemplate, newNodeRep))

        return newCtxt, updates

    def _generateIngressPointerUpdates(
            self, nodeMemoryConstraint: NodeMemoryConstraint, tilingSchedule: TilingSchedule, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, List[CodeSnippet]]:

        updates = []
        newCtxt = ctxt.copy()

        updateDict = self._generatePointerUpdates(ctxt, operatorRepresentation, tilingSchedule.inputLoadSchedule,
                                                  nodeMemoryConstraint, tilingSchedule)

        for key, updateList in updateDict.items():

            newCtxt, newNodeRep = self._hoistDMAUpdates(newCtxt, key, updateList, operatorRepresentation)
            updates.append(CodeSnippet(self._updateDMATransferStructTemplate, newNodeRep))

        return newCtxt, updates

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

    def _generateDMACode(self, nodeMemoryConstraint: NodeMemoryConstraint, ctxt: NetworkContext,
                         operatorRepresentation: OperatorRepresentation, loadSchedule: List[Dict[str, HyperRectangle]],
                         direction: Literal["ToL1", "FromL1"]) -> Tuple[List[CodeSnippet], List[CodeSnippet]]:

        DMATransferCalls = []
        DMAWaitStatements = []
        transferNodeRep = {}

        loadStep = loadSchedule[0]

        for idx, (key, rectangle) in enumerate(loadStep.items()):

            permName = f"in{idx}_perm"

            externalPtr = ctxt.lookup(ctxt.lookup(operatorRepresentation[key])._referenceName)
            internalPtr = ctxt.lookup(operatorRepresentation[key])
            assert internalPtr._memoryLevel == self.localMemory

            tensorName = key
            dmaName = self._DMAStructName(tensorName)

            transferNodeRep = {
                **transferNodeRep,
                **{
                    'innerTilePtr': internalPtr.name,
                    "outerTilePtr": externalPtr.name,
                    "stateReference": dmaName
                }
            }

            tensorMemoryConstraint = nodeMemoryConstraint.tensorMemoryConstraints[internalPtr._referenceName]
            finalMemoryLevel = self.isFinalMemoryLevel(tensorMemoryConstraint)
            struct = self._rectToDMAStruct(ctxt, rectangle, direction, internalPtr.name, externalPtr.name,
                                           finalMemoryLevel)

            transferNodeRep["stateStruct"] = struct
            _ = ctxt.hoistStruct(struct, dmaName, Snitch_DMA_copy)
            ctxt.lookup(dmaName)._users += [operatorRepresentation['nodeName']]

            if permName in operatorRepresentation and direction == "ToL1":

                DMATransferCalls.append(CodeSnippet(self._iteratedMoveTileInTemplate, transferNodeRep))
            else:
                DMATransferCalls.append(CodeSnippet(self._moveTileInTemplate, transferNodeRep))

            DMAWaitStatements.append(CodeSnippet(self._blockTileInTemplate, transferNodeRep))

        return DMATransferCalls, DMAWaitStatements

    def _generateIngressDMACode(
            self, tilingSchedule: TilingSchedule, nodeMemoryConstraint: NodeMemoryConstraint, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[List[CodeSnippet], List[CodeSnippet]]:

        importLoadStep = tilingSchedule.inputLoadSchedule
        ingressDMATransferCalls, ingressDMAWaitStatements = self._generateDMACode(nodeMemoryConstraint, ctxt,
                                                                                  operatorRepresentation,
                                                                                  importLoadStep, "ToL1")
        return ingressDMATransferCalls, ingressDMAWaitStatements

    def _generateEgressDMACode(
            self, tilingSchedule: TilingSchedule, nodeMemoryConstraint: NodeMemoryConstraint, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[List[CodeSnippet], List[CodeSnippet]]:

        exportLoadStep = tilingSchedule.outputLoadSchedule
        egressDMATransferCalls, egressDMAWaitStatements = self._generateDMACode(nodeMemoryConstraint, ctxt,
                                                                                operatorRepresentation, exportLoadStep,
                                                                                "FromL1")

        return egressDMATransferCalls, egressDMAWaitStatements

    def _tilingLoop(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
                    nodeMemoryConstraint: NodeMemoryConstraint, tilingSchedule: TilingSchedule,
                    variableReplacement: VariableReplacementScheme,
                    operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, ExecutionBlock, bool]:

        ingressDMATransferCalls, ingressDMAWaitStatements = self._generateIngressDMACode(
            tilingSchedule, nodeMemoryConstraint, ctxt, operatorRepresentation)

        egressDMATransferCalls, egressDMAWaitStatements = self._generateEgressDMACode(
            tilingSchedule, nodeMemoryConstraint, ctxt, operatorRepresentation)

        ctxt, ingressDMAUpdates = self._generateIngressPointerUpdates(nodeMemoryConstraint, tilingSchedule, ctxt,
                                                                      operatorRepresentation)
        ctxt, egressDMAUpdates = self._generateEgressPointerUpdates(nodeMemoryConstraint, tilingSchedule, ctxt,
                                                                    operatorRepresentation)

        openLoopStatement = [CodeSnippet(self._openTileLoopTemplate, {**operatorRepresentation})]

        closeLoopStatement = [CodeSnippet(self._closeTileLoopTemplate, {**operatorRepresentation})]

        variableUpdates = self._generateVariableUpdates(tilingSchedule, variableReplacement, ctxt,
                                                        operatorRepresentation)

        metaInfo = TilingMetaInfo(nodeName = operatorRepresentation['nodeName'] + "_L2",
                                  nodeOps = operatorRepresentation['nodeOps'],
                                  numTiles = len(tilingSchedule.outputLoadSchedule),
                                  tileIdxVar = "TILING_I",
                                  kernelLevelTiling = True)

        newExecutionBlock = self.generateAllTilingCode(executionBlock, metaInfo, ingressDMATransferCalls,
                                                       ingressDMAWaitStatements, ingressDMAUpdates,
                                                       egressDMATransferCalls, egressDMAWaitStatements,
                                                       egressDMAUpdates, variableUpdates, openLoopStatement,
                                                       closeLoopStatement, [], [])

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

        numTiles, tileIdxPtr = self._hoistTileNumAndIdxPtr(ctxt, tilingSchedules)
        operatorRepresentation["numTiles"] = numTiles.name
        operatorRepresentation["tileIdxPtr"] = tileIdxPtr.name

        return self._tilingLoop(ctxt, executionBlock, nodeMemoryConstraint, flatTilingSchedule, variableReplacement,
                                operatorRepresentation)


class SnitchClusterTilingGenerationSB(SnitchClusterTilingSB, SingleBufferingTilingMixIn):
    pass
