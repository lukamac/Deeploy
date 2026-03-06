# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import SequentialPass
from Deeploy.DeeployTypes import NetworkContext, VariableBuffer
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel


class AnnotateDefaultMemoryLevel(SequentialPass):

    def __init__(self, memoryHierarchy: MemoryHierarchy):
        super().__init__()
        self.memoryHierarchy = memoryHierarchy

    def apply(self, ctxt: NetworkContext, graph: gs.Graph) -> Tuple[NetworkContext, gs.Graph]:
        for _buffer in {**ctxt.localObjects, **ctxt.globalObjects}.values():
            if not hasattr(_buffer, "_memoryLevel"):
                _buffer._memoryLevel = self.memoryHierarchy.getDefaultMemoryLevel().name
        return ctxt, graph


class AnnotateIOMemoryLevel(SequentialPass):

    def __init__(self, ioLevel: str):
        super().__init__()
        self.ioLevel = ioLevel

    def apply(self, ctxt: NetworkContext, graph: gs.Graph) -> Tuple[NetworkContext, gs.Graph]:
        buffers = []

        def globalBuffers(tensors: List[gs.Tensor]) -> List[VariableBuffer]:
            return [ctxt.globalObjects[tensor.name] for tensor in tensors if tensor.name in ctxt.globalObjects.keys()]

        inputBuffers = globalBuffers(graph.inputs)
        buffers += filter(lambda _buffer: isinstance(_buffer, ctxt.VariableBuffer) and len(_buffer._users) > 0,
                          inputBuffers)

        outputBuffers = globalBuffers(graph.outputs)
        buffers += filter(lambda _buffer: isinstance(_buffer, ctxt.VariableBuffer), outputBuffers)

        for _buffer in buffers:
            _buffer._memoryLevel = self.ioLevel

        return ctxt, graph


class AnnotateNeurekaWeightMemoryLevel(SequentialPass):

    def __init__(self, neurekaEngineName: str, weightMemoryLevel: MemoryLevel):
        self._weightMemoryLevel = weightMemoryLevel
        self.neurekaEngineName = neurekaEngineName
        super().__init__()

    def apply(self, ctxt: NetworkContext, graph: gs.Graph) -> Tuple[NetworkContext, gs.Graph]:
        weightMemoryOccupation = 0

        def inWmem(buffer: VariableBuffer) -> bool:
            return hasattr(buffer, "_memoryLevel") and buffer._memoryLevel == self._weightMemoryLevel.name

        # Current weight memory occupation
        for buffer in {**ctxt.globalObjects, **ctxt.localObjects}.values():
            if isinstance(buffer, VariableBuffer) and inWmem(buffer):
                weightMemoryOccupation += buffer.sizeInBytes()

        neurekaNodes = [node for node in graph.nodes if node.attrs["engine"] == self.neurekaEngineName]
        for node in neurekaNodes:
            if node.op in ["Conv", "RequantizedConv"] and ctxt.is_buffer(node.inputs[1].name):
                buffer = ctxt.lookup(node.inputs[1].name)
                assert isinstance(buffer, VariableBuffer)
                size = buffer.sizeInBytes()
                if not inWmem(buffer) and weightMemoryOccupation + size < self._weightMemoryLevel.size:
                    buffer._memoryLevel = self._weightMemoryLevel.name
                    weightMemoryOccupation += size

        return ctxt, graph
