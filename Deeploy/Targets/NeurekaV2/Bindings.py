# ----------------------------------------------------------------------
#
# File: NeurekaV2Bindings.py
#
# Last edited: 10.07.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author:
# Luka Macan, University of Bologna
# Moritz Scherer, ETH Zurich
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

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import int8_t, int32_t, uint8_t
from Deeploy.DeeployTypes import NodeBinding
from Deeploy.MemoryLevelExtension.MemoryLevels import NodeMemoryLevelChecker, memoryAwareNodeBindingExtension
from Deeploy.Targets.Generic.TypeCheckers import ConvChecker
from Deeploy.Targets.NeurekaV2.Templates.ConvTemplate import NeurekaV2DenseConv2D_Template, \
    NeurekaV2DWConv2D_Template, NeurekaV2PWConv2D_Template, NeurekaV2RqntDenseConv2D_Template, \
    NeurekaV2RqntDWConv2D_Template, NeurekaV2RqntPWConv2D_Template
from Deeploy.Targets.PULPOpen.Bindings import ClusterTransformer
from Deeploy.Targets.PULPOpen.TypeCheckers import PULPConvChecker

NeurekaV2RQSPWConv2DBindings = [
    NodeBinding(
        PULPConvChecker(
            [PointerClass(data_in_type),
             PointerClass(weight_type),
             PointerClass(int32_t),
             PointerClass(int32_t)], [PointerClass(data_out_type)]), NeurekaV2RqntPWConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for data_out_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]
NeurekaV2PWConv2DBindings = [
    NodeBinding(
        ConvChecker(
            [PointerClass(data_in_type), PointerClass(weight_type),
             PointerClass(int32_t)], [PointerClass(int32_t)]), NeurekaV2PWConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]

NeurekaV2WmemRQSPWConv2DBindings = [
    memoryAwareNodeBindingExtension(binding, NodeMemoryLevelChecker([None, "WeightMemory_SRAM", None, None], [None]))
    for binding in NeurekaV2RQSPWConv2DBindings
]
NeurekaV2WmemPWConv2DBindings = [
    memoryAwareNodeBindingExtension(binding, NodeMemoryLevelChecker([None, "WeightMemory_SRAM"], [None]))
    for binding in NeurekaV2PWConv2DBindings
]

NeurekaV2RQSDWConv2DBindings = [
    NodeBinding(
        PULPConvChecker(
            [PointerClass(data_in_type),
             PointerClass(weight_type),
             PointerClass(int32_t),
             PointerClass(int32_t)], [PointerClass(data_out_type)]), NeurekaV2RqntDWConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for data_out_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]
NeurekaV2DWConv2DBindings = [
    NodeBinding(
        ConvChecker(
            [PointerClass(data_in_type), PointerClass(weight_type),
             PointerClass(int32_t)], [PointerClass(int32_t)]), NeurekaV2DWConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]

NeurekaV2WmemRQSDWConv2DBindings = [
    memoryAwareNodeBindingExtension(binding, NodeMemoryLevelChecker([None, "WeightMemory_SRAM", None, None], [None]))
    for binding in NeurekaV2RQSDWConv2DBindings
]
NeurekaV2WmemDWConv2DBindings = [
    memoryAwareNodeBindingExtension(binding, NodeMemoryLevelChecker([None, "WeightMemory_SRAM"], [None]))
    for binding in NeurekaV2DWConv2DBindings
]

NeurekaV2RQSDenseConv2DBindings = [
    NodeBinding(
        PULPConvChecker(
            [PointerClass(data_in_type),
             PointerClass(weight_type),
             PointerClass(int32_t),
             PointerClass(int32_t)], [PointerClass(data_out_type)]), NeurekaV2RqntDenseConv2D_Template,
        ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for data_out_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]
NeurekaV2DenseConv2DBindings = [
    NodeBinding(
        ConvChecker(
            [PointerClass(data_in_type), PointerClass(weight_type),
             PointerClass(int32_t)], [PointerClass(int32_t)]), NeurekaV2DenseConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]

NeurekaV2WmemRQSDenseConv2DBindings = [
    memoryAwareNodeBindingExtension(binding, NodeMemoryLevelChecker([None, "WeightMemory_SRAM", None, None], [None]))
    for binding in NeurekaV2RQSDenseConv2DBindings
]
NeurekaV2WmemDenseConv2DBindings = [
    memoryAwareNodeBindingExtension(binding, NodeMemoryLevelChecker([None, "WeightMemory_SRAM"], [None]))
    for binding in NeurekaV2DenseConv2DBindings
]
