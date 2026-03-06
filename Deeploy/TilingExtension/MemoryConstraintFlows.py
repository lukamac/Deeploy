# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, OrderedDict, Set

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, ONNXLayer, VariableBuffer
from Deeploy.TilingExtension.GenericFlow import GenericFlow


class InnerBufferLivenessAnalysis(GenericFlow[str, gs.Node]):

    def __init__(self, ctxt: NetworkContext, steps: List[gs.Node]) -> None:
        self.ctxt = ctxt
        self.steps = steps

    def initLive(self) -> Set[str]:
        return set(tensor.name
                   for node in self.steps
                   for tensor in node.inputs
                   if len(tensor.inputs) == 0 or tensor.inputs[0] not in self.steps)

    def computeGen(self, step: gs.Node) -> Set[str]:
        return set(tensor.name for tensor in step.outputs)

    def computeKill(self, step: gs.Node) -> Set[str]:

        def isKill(name: str) -> bool:
            buffer = self.ctxt.lookup(name)
            assert isinstance(buffer, VariableBuffer)
            patternUsers = [node.name for node in self.steps if node.name in buffer._users]

            # Check that our steps are correctly ordered compared to the global buffer user ordering
            last_global_index = -1
            for user in patternUsers:
                global_index = buffer._users.index(user)
                assert last_global_index < global_index, (f"Buffer {buffer.name}'s user {user} uses the buffer sooner"
                                                          " in the flow's steps compared to the buffer's user list.")
                last_global_index = global_index

            return buffer.name == patternUsers[-1]

        return set(tensor.name for tensor in step.inputs if isKill(tensor.name))


class OuterBufferLivenessAnalysis(GenericFlow[str, List[gs.Node]]):

    def __init__(self, ctxt: NetworkContext) -> None:
        self.ctxt = ctxt

    def initLive(self, layerBinding: OrderedDict[str, ONNXLayer]) -> Set[str]:
        globalInputTensors = {
            name for name, obj in self.ctxt.globalObjects.items()
            if isinstance(obj, self.ctxt.VariableBuffer) and len(obj._users) > 0
        }
        producedTensors = {tensor.name for layer in layerBinding.values() for tensor in layer.node.outputs}
        return globalInputTensors - producedTensors

    def computeGen(self, step: List[gs.Node]) -> Set[str]:
        return set(tensor.name
                   for node in step
                   for tensor in node.outputs
                   if len(tensor.outputs) == 0 or any(user not in step for user in tensor.outputs))

    def computeKill(self, step: List[gs.Node]) -> Set[str]:
        nodeNames = [node.name for node in step]

        def isKill(tensorName: str) -> bool:
            buffer = self.ctxt.lookup(tensorName)
            assert isinstance(buffer, VariableBuffer)
            return not isinstance(buffer, ConstantBuffer) and len(buffer._users) > 0 and buffer._users[-1] in nodeNames

        return set(tensor.name for node in step for tensor in node.inputs if isKill(tensor.name))
