# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.CodeTransformationPasses.Closure import ClosureGeneration, MemoryAwareClosureGeneration
from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import ArgumentStructGeneration, \
    MemoryManagementGeneration
from Deeploy.CommonExtensions.DataTypes import float32_t, int8_t, int32_t, uint8_t
from Deeploy.DeeployTypes import CodeTransformation, NodeBinding, NodeTypeChecker
from Deeploy.FutureExtension.CodeTransformationPasses.FutureCodeTransformation import FutureGeneration
from Deeploy.Targets.Generic.Templates import FloatReduceSumTemplate, iNoNormTemplate
from Deeploy.Targets.Generic.TypeCheckers import AddChecker, GEMMChecker, RQAddChecker, ReduceMeanChecker, SoftmaxChecker, iNoNormChecker
from Deeploy.Targets.Snitch.CodeTransformationPasses import SnitchClusterTiling, SnitchCoreFilterPass, \
    SnitchProfileExecutionBlockPass, SnitchSynchCoresPass
from Deeploy.Targets.Snitch.DMA.SnitchDma import SnitchDma
from Deeploy.Targets.Snitch.Templates import AddTemplate, FloatGemmTemplate, RQAddTemplate, iSoftmaxTemplate
from Deeploy.Targets.Snitch.Templates.FloatAddTemplate import multiCoreFloatAddTemplate
from Deeploy.Targets.Snitch.Templates.FloatAddTemplate import referenceTemplate as FloatAdd_Template
from Deeploy.Targets.Snitch.Templates.FloatAddTemplate import singleCoreFloatAddTemplate
from Deeploy.Targets.Snitch.Templates.FloatConvTemplate import parallelTemplate as FloatConv_Template_Parallel
from Deeploy.Targets.Snitch.Templates.FloatFusedAddReluTemplate import referenceTemplate as FloatFusedAddRelu_Template
from Deeploy.Targets.Snitch.Templates.FloatFusedConvReluTemplate import referenceTemplate as FloatFusedConvRelu_Template
from Deeploy.Targets.Snitch.Templates.FloatSoftmaxTemplate import FloatSoftmax_Template
from Deeploy.Targets.Snitch.Templates.GemmTemplate import SnitchGemm_Template
from Deeploy.Targets.Snitch.Templates.RqGemmTemplate import SnitchRqGemm_Template
from Deeploy.TilingExtension.CodeTransformationPasses.TilingVariableReplacement import TilingVariableReplacement, \
    TilingVariableReplacementUpdate

TilingCallClosure = partial(ClosureGeneration, closureSuffix = "_tiling_closure")
MemoryAwareFunctionCallClosure = partial(MemoryAwareClosureGeneration,
                                         closureSuffix = "_closure",
                                         startRegion = "L2",
                                         endRegion = "L1")

BasicTransformer = CodeTransformation([
    SnitchSynchCoresPass(),
    ArgumentStructGeneration(),
    MemoryManagementGeneration(),
    FutureGeneration(),
])

BasicComputeTransformer = CodeTransformation([
    SnitchCoreFilterPass("compute"),
    SnitchSynchCoresPass(),
    ArgumentStructGeneration(),
    MemoryManagementGeneration(),
])

TiledTransformer = CodeTransformation([
    SnitchCoreFilterPass("compute"),
    SnitchProfileExecutionBlockPass(),
    TilingVariableReplacement("L1"),
    TilingCallClosure(writeback = False),
    SnitchSynchCoresPass(),
    TilingVariableReplacementUpdate("L1"),
    SnitchClusterTiling("L2", "L1", SnitchDma()),
    ArgumentStructGeneration(),
    MemoryManagementGeneration("L1"),
    MemoryAwareFunctionCallClosure(writeback = False, generateStruct = True),
    MemoryManagementGeneration()
])

SnitchiSoftmaxBindings = [
    NodeBinding(SoftmaxChecker([PointerClass(_type)], [PointerClass(uint8_t)]), iSoftmaxTemplate.referenceTemplate,
                TiledTransformer) for _type in [int8_t, uint8_t]
] + [
    NodeBinding(SoftmaxChecker([PointerClass(float32_t)], [PointerClass(float32_t)]), FloatSoftmax_Template,
                TiledTransformer)
]

SnitchiNoNormBindings = [
    NodeBinding(
        iNoNormChecker([PointerClass(_type), PointerClass(int8_t),
                        PointerClass(int32_t)], [PointerClass(int8_t)]), iNoNormTemplate.referenceTemplate,
        TiledTransformer) for _type in [int8_t]
]
SnitchRQAddBindings = [
    NodeBinding(RQAddChecker([PointerClass(_type), PointerClass(_type)], [PointerClass(_type)]),
                RQAddTemplate.referenceTemplate, TiledTransformer) for _type in [int8_t]
]
SnitchAddBindings = [
    NodeBinding(AddChecker([PointerClass(_type), PointerClass(_type)], [PointerClass(int32_t)]),
                AddTemplate.referenceTemplate, TiledTransformer) for _type in [int8_t]
]
SnitchFloatAddBinding = NodeBinding(
    NodeTypeChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]), FloatAdd_Template,
    TiledTransformer)
SnitchSingleCoreFloatAddBinding = NodeBinding(
    NodeTypeChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
    singleCoreFloatAddTemplate, BasicTransformer)
SnitchMultiCoreFloatAddBinding = NodeBinding(
    NodeTypeChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
    multiCoreFloatAddTemplate, BasicComputeTransformer)
SnitchGemmBindings = [
    NodeBinding(
        GEMMChecker([PointerClass(int8_t), PointerClass(int8_t),
                     PointerClass(int32_t)], [PointerClass(int32_t)]), SnitchGemm_Template, TiledTransformer)
] + [
    NodeBinding(
        GEMMChecker([PointerClass(float32_t), PointerClass(float32_t),
                     PointerClass(float32_t)], [PointerClass(float32_t)]), FloatGemmTemplate.referenceTemplate,
        TiledTransformer)
]
SnitchRqGemmBindings = [
    NodeBinding(
        GEMMChecker([
            PointerClass(int8_t),
            PointerClass(int8_t),
            PointerClass(int32_t),
            PointerClass(int32_t),
            PointerClass(int32_t)
        ], [PointerClass(int8_t)]), SnitchRqGemm_Template, TiledTransformer)
]
SnitchFloatConvBinding = NodeBinding(
    NodeTypeChecker([PointerClass(float32_t), PointerClass(float32_t),
                     PointerClass(float32_t)], [PointerClass(float32_t)]), FloatConv_Template_Parallel,
    BasicComputeTransformer)
SnitchFloatFusedAddReluBinding = NodeBinding(
    NodeTypeChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
    FloatFusedAddRelu_Template, BasicComputeTransformer)
SnitchFloatFusedConvReluBinding = NodeBinding(
    NodeTypeChecker([PointerClass(float32_t), PointerClass(float32_t),
                     PointerClass(float32_t)], [PointerClass(float32_t)]), FloatFusedConvRelu_Template,
    BasicComputeTransformer)

SnitchFloatReduceSumBindings = [
    NodeBinding(ReduceMeanChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatReduceSumTemplate.referenceTemplate, BasicComputeTransformer)
]
