# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import numpy as np

from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import \
    RemoveEmptyConvBiasPass
from Deeploy.DeeployTypes import ConstantBuffer, DeploymentEngine, DeploymentPlatform, NodeMapper, NodeTemplate, ONNXLayer, \
    StructBuffer, TopologyOptimizer, TransientBuffer, VariableBuffer
from Deeploy.Targets.Generic.Bindings import BasicGatherBindings, BasicLayerNormBindings, BasicMatMulBindings, \
    BasicPad1DBindings, BasicPad2DBindings, BasicReshapeBindings, BasicRQIntegerDivBinding, BasicConv2DBindings, BasicGEMMBindings
from Deeploy.Targets.Generic.Layers import AddLayer, GatherLayer, GEMMLayer, LayerNormLayer, MatMulLayer, PadLayer, \
    ReshapeLayer, RQGEMMLayer, RQIntegerDivLayer, SoftmaxLayer, TransposeLayer, iNoNormLayer
from Deeploy.Targets.Generic.Parsers import AddParser, Conv2DParser, GatherParser, GenericConv2DParser, GenericFusedConv2DReluParser, MatMulParser, Pad1DParser, Pad2DParser, \
    RQAddParser, RQIntegerDivParser, SoftmaxParser, UnsqueezeParser, iLayerNormParser, iNoNormParser, iSoftmaxParser
from Deeploy.Targets.Generic.Platform import AvgPoolMapper, ReshapeMapper, TransposeMapper
from Deeploy.Targets.Generic.Templates import AllocateTemplate as BasicAllocateTemplate
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import AddRequantMergePass, GEMMRequantMergePass, \
    IntegerDivRequantMergePass, MergeConstAddAndRequantPass, MergeTrueIntegerDivRequantShiftPass, RQSSplitPass, \
    SkipEmptyConcatPass, SkipUnityRequantPass, iGELURequantMergePass, iHardswishRequantMergePass, ExtractPaddingFromPoolPass, ExtractPaddingFromConvPass, MatMulAddMergePass, MergeGemmAddPass
from Deeploy.Targets.PULPOpen.Platform import RQAddMapper
from Deeploy.Targets.Snitch.Bindings import SnitchFloatAddBinding, SnitchFloatFusedAddReluBinding, SnitchSingleCoreFloatAddBinding, SnitchMultiCoreFloatAddBinding, SnitchFloatConvBinding, SnitchFloatFusedConvReluBinding
from Deeploy.Targets.Snitch.Layers import ConvLayer
from Deeploy.Targets.Snitch.Parsers import SnitchGEMMParser, SnitchRQGEMMParser
from Deeploy.Targets.Snitch.Templates import AllocateTemplate, FreeTemplate
from Deeploy.Targets.Snitch.Tiler import SnitchAddTileReadyBindings, SnitchFloatFusedAddReluTileReadyBindings, SnitchFloatFusedConv2dReluTileReadyBindings, \
    SnitchGemmTilingReadyBindings, SnitchiNoNormTilingReadyBindings, SnitchiSoftmaxTilingReadyBindings, SnitchRQAddTilingReadyBindings, \
    SnitchRqGemmTilingReadyBindings

GatherMapper = NodeMapper(GatherParser(), BasicGatherBindings)
Pad1DMapper = NodeMapper(Pad1DParser(), BasicPad1DBindings)
Pad2DMapper = NodeMapper(Pad2DParser(), BasicPad2DBindings)
UnsqueezeMapper = NodeMapper(UnsqueezeParser(), BasicReshapeBindings)

RQIntegerDivMapper = NodeMapper(RQIntegerDivParser(), [BasicRQIntegerDivBinding])

MatMulMapper = NodeMapper(MatMulParser(), BasicMatMulBindings)
GemmMapper = NodeMapper(SnitchGEMMParser(), BasicGEMMBindings + SnitchGemmTilingReadyBindings)
RqGemmMapper = NodeMapper(SnitchRQGEMMParser(), SnitchRqGemmTilingReadyBindings)
iSoftmaxMapper = NodeMapper(iSoftmaxParser(), SnitchiSoftmaxTilingReadyBindings)
SoftmaxMapper = NodeMapper(SoftmaxParser(), SnitchiSoftmaxTilingReadyBindings)
iNoNormMapper = NodeMapper(iNoNormParser(), SnitchiNoNormTilingReadyBindings)
iLayerNormMapper = NodeMapper(iLayerNormParser(), BasicLayerNormBindings)
RQAddMapper = NodeMapper(RQAddParser(), SnitchRQAddTilingReadyBindings)

testAdderBindings = [
    SnitchMultiCoreFloatAddBinding,
    #SnitchSingleCoreFloatAddBinding, 
]

AddMapper = NodeMapper(AddParser(), testAdderBindings + SnitchAddTileReadyBindings + [SnitchFloatAddBinding])
FloatFusedAddReluMapper = NodeMapper(AddParser(), [SnitchFloatFusedAddReluBinding] + SnitchFloatFusedAddReluTileReadyBindings)
FloatFusedConv2DReluMapper = NodeMapper(GenericFusedConv2DReluParser(), [SnitchFloatFusedConvReluBinding] + SnitchFloatFusedConv2dReluTileReadyBindings)
Conv2DMapper = NodeMapper(GenericConv2DParser(), BasicConv2DBindings + [SnitchFloatConvBinding])

SnitchMapping = {
    'Conv': ConvLayer([Conv2DMapper]),
    'RQIntegerDiv': RQIntegerDivLayer([RQIntegerDivMapper]),
    'Gather': GatherLayer([GatherMapper]),
    'Pad': PadLayer([Pad1DMapper, Pad2DMapper]),
    'Unsqueeze': ReshapeLayer([UnsqueezeMapper]),
    'MatMul': MatMulLayer([MatMulMapper]),
    'Gemm': GEMMLayer([GemmMapper]),
    'RQGemm': RQGEMMLayer([RqGemmMapper]),
    'iSoftmax': SoftmaxLayer([iSoftmaxMapper]),
    'Softmax': SoftmaxLayer([SoftmaxMapper]),
    'iNoNorm': iNoNormLayer([iNoNormMapper]),
    'iLayerNorm': LayerNormLayer([iLayerNormMapper]),
    'RequantizedAdd': AddLayer([RQAddMapper]),
    'Add': AddLayer([AddMapper]),
    'Transpose': TransposeLayer([TransposeMapper]),
    'FusedConvRelu': ConvLayer([FloatFusedConv2DReluMapper]),
    'FusedAddRelu': AddLayer([FloatFusedAddReluMapper]),
    'AveragePool': ONNXLayer([AvgPoolMapper]),
    'Reshape': ReshapeLayer([ReshapeMapper]),
}


class SnitchVariableBuffer(VariableBuffer):

    initTemplate = AllocateTemplate.snitchL2InitTemplate
    allocTemplate = AllocateTemplate.snitchGenericAllocate
    deallocTemplate = FreeTemplate.snitchGenericFree

    def _bufferRepresentation(self):

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        return {
            "type": self._instance,
            "name": self.name,
            "size": int(np.prod(self.shape)),
            "_memoryLevel": memoryLevel
        }


class SnitchTransientBuffer(TransientBuffer):

    initTemplate = AllocateTemplate.snitchL2InitTemplate
    allocTemplate = AllocateTemplate.snitchGenericAllocate
    deallocTemplate = FreeTemplate.snitchGenericFree

    # allocTemplate = AllocateTemplate.snitchL2AllocateTemplate
    # deallocTemplate = FreeTemplate.snitchL2GlobalTemplate

    def _bufferRepresentation(self):

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        return {"type": self._type, "name": self.name, "size": self.size, "_memoryLevel": memoryLevel}


class SnitchConstantBuffer(ConstantBuffer):

    initTemplate = AllocateTemplate.snitchGenericGlobalInitTemplate
    allocTemplate = AllocateTemplate.snitchL2GlobalAllocateTemplate
    deallocTemplate = FreeTemplate.snitchL2GlobalTemplate

    def _bufferRepresentation(self):
        operatorRepresentation = super()._bufferRepresentation()

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        operatorRepresentation["_memoryLevel"] = memoryLevel

        return operatorRepresentation


class SnitchStructBuffer(StructBuffer):

    initTemplate = BasicAllocateTemplate.referenceStructInitTemplate
    allocTemplate = BasicAllocateTemplate.referenceStructAllocateTemplate
    deallocTemplate = NodeTemplate("")


SnitchOptimizer = TopologyOptimizer([
    SkipEmptyConcatPass(),
    SkipUnityRequantPass(previous_op_regex = "Concat", num_inputs = 2),
    SkipUnityRequantPass(previous_op_regex = "Reshape|Transpose", num_inputs = 1),
    SkipUnityRequantPass(previous_op_regex = "Reshape|Transpose", num_inputs = 1),
    RQSSplitPass(),
    MergeTrueIntegerDivRequantShiftPass(),
    IntegerDivRequantMergePass(),
    iGELURequantMergePass(),
    iHardswishRequantMergePass(),
    MergeConstAddAndRequantPass(),
    AddRequantMergePass(),
    GEMMRequantMergePass(),
    MatMulAddMergePass(),
    MergeGemmAddPass(),
    MergeConstAddAndRequantPass(),
    ExtractPaddingFromConvPass(),
    ExtractPaddingFromPoolPass(),
    RemoveEmptyConvBiasPass(),
])

_includeList = [
    "snrt.h",
    "DeeploySnitchMath.h",
]


class SnitchClusterEngine(DeploymentEngine):

    def __init__(self, name: str, Mapping = SnitchMapping, initCode = "", includeList = _includeList) -> None:
        super().__init__(name, Mapping, initCode, includeList)


class SnitchPlatform(DeploymentPlatform):

    def __init__(self,
                 engines = [SnitchClusterEngine("SnitchCluster")],
                 variableBuffer = SnitchVariableBuffer,
                 constantBuffer = SnitchConstantBuffer,
                 structBuffer = SnitchStructBuffer,
                 transientBuffer = SnitchTransientBuffer,
                 includeList: List[str] = _includeList):
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer, transientBuffer)
