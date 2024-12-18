# ----------------------------------------------------------------------
#
# File: Platform.py
#
# Last edited: 26.07.2024
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

from typing import Optional

import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import \
    RequantizedGemmToPwPass
from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, NodeTemplate, TopologyOptimizer
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.Targets.NeurekaV2.Engine import NeurekaV2Engine
from Deeploy.Targets.NeurekaV2.Templates.AllocateTemplate import neurekaGenericGlobalInitTemplate
from Deeploy.Targets.PULPOpen.Platform import MemoryPULPPlatform, MemoryPULPPlatformWrapper, PULPClusterEngine, \
    PULPOptimizer, PULPPlatform, PULPStructBuffer, PULPTransientBuffer, PULPVariableBuffer

NeurekaV2Optimizer = TopologyOptimizer([
    *PULPOptimizer.passes,
    RequantizedGemmToPwPass(),
])


class NeurekaV2ConstantBuffer(ConstantBuffer):

    initTemplate = neurekaGenericGlobalInitTemplate
    allocTemplate = NodeTemplate("")
    deallocTemplate = NodeTemplate("")

    def _bufferRepresentation(self):
        operatorRepresentation = super()._bufferRepresentation()
        operatorRepresentation["_memoryLevel"] = getattr(self, "_memoryLevel", None)
        return operatorRepresentation


class NeurekaV2Platform(PULPPlatform):

    def __init__(self,
                 engines = [NeurekaV2Engine("NeurekaV2"), PULPClusterEngine("PULPCluster")],
                 variableBuffer = PULPVariableBuffer,
                 constantBuffer = NeurekaV2ConstantBuffer,
                 structBuffer = PULPStructBuffer,
                 transientBuffer = PULPTransientBuffer) -> None:
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer, transientBuffer)


class MemoryNeurekaV2Platform(MemoryPULPPlatform):

    def __init__(self,
                 memoryHierarchy: MemoryHierarchy,
                 defaultTargetMemoryLevel: MemoryLevel,
                 weightMemoryLevel: Optional[MemoryLevel] = None,
                 engines = [NeurekaV2Engine("NeurekaV2"), PULPClusterEngine("PULPCluster")],
                 variableBuffer = PULPVariableBuffer,
                 constantBuffer = NeurekaV2ConstantBuffer,
                 structBuffer = PULPStructBuffer,
                 transientBuffer = PULPTransientBuffer) -> None:
        super().__init__(memoryHierarchy, defaultTargetMemoryLevel, engines, variableBuffer, constantBuffer,
                         structBuffer, transientBuffer)
        self.weightMemoryLevel = weightMemoryLevel

    def getTargetMemoryLevel(self, node: gs.Node, tensorName: str, ctxt: NetworkContext) -> str:
        if self.weightMemoryLevel is not None and ctxt.lookup(tensorName)._memoryLevel == self.weightMemoryLevel.name:
            return self.weightMemoryLevel.name
        return super().getTargetMemoryLevel(node, tensorName, ctxt)


class MemoryNeurekaV2PlatformWrapper(MemoryPULPPlatformWrapper):

    def __init__(self,
                 platform: NeurekaV2Platform,
                 memoryHierarchy: MemoryHierarchy,
                 defaultTargetMemoryLevel: MemoryLevel,
                 weightMemoryLevel: Optional[MemoryLevel] = None):
        assert isinstance(platform, NeurekaV2Platform), \
        f"Given platform is not an instance of NeurekaV2Platform. Platform type: {type(platform).__name__}"
        super().__init__(platform, memoryHierarchy, defaultTargetMemoryLevel)
        self.weightMemoryLevel = weightMemoryLevel

    def getTargetMemoryLevel(self, node: gs.Node, tensorName: str, ctxt: NetworkContext) -> str:
        if self.weightMemoryLevel is not None and ctxt.lookup(tensorName)._memoryLevel == self.weightMemoryLevel.name:
            return self.weightMemoryLevel.name
        return super().getTargetMemoryLevel(node, tensorName, ctxt)
