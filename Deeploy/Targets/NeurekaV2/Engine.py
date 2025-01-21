# ----------------------------------------------------------------------
#
# File: Engine.py
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

from typing import List

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import DeploymentEngine, NodeMapper
from Deeploy.Targets.Generic.Layers import ConvLayer
from Deeploy.Targets.Neureka.Parsers import NeurekaDenseConv2DParser, NeurekaDWConv2DParser, NeurekaPWConv2DParser, \
    NeurekaRQSDenseConv2DParser, NeurekaRQSDWConv2DParser, NeurekaRQSPWConv2DParser
from Deeploy.Targets.NeurekaV2.Tiler import NeurekaV2DenseConv2DTilingReadyBindings, \
    NeurekaV2DWConv2DTilingReadyBindings, NeurekaV2PWConv2DTilingReadyBindings, \
    NeurekaV2RQSDenseConv2DTilingReadyBindings, NeurekaV2RQSDWConv2DTilingReadyBindings, \
    NeurekaV2RQSPWConv2DTilingReadyBindings, NeurekaV2WmemDenseConv2DTilingReadyBindings, \
    NeurekaV2WmemDWConv2DTilingReadyBindings, NeurekaV2WmemPWConv2DTilingReadyBindings, \
    NeurekaV2WmemRQSDenseConv2DTilingReadyBindings, NeurekaV2WmemRQSDWConv2DTilingReadyBindings, \
    NeurekaV2WmemRQSPWConv2DTilingReadyBindings
from Deeploy.Targets.PULPOpen.Layers import PULPRQSConvLayer

NeurekaV2RqntPWConv2DMapper = NodeMapper(
    NeurekaRQSPWConv2DParser(), NeurekaV2WmemRQSPWConv2DTilingReadyBindings + NeurekaV2RQSPWConv2DTilingReadyBindings)
NeurekaV2PWConv2DMapper = NodeMapper(NeurekaPWConv2DParser(),
                                     NeurekaV2WmemPWConv2DTilingReadyBindings + NeurekaV2PWConv2DTilingReadyBindings)

NeurekaV2RqntDWConv2DMapper = NodeMapper(
    NeurekaRQSDWConv2DParser(), NeurekaV2WmemRQSDWConv2DTilingReadyBindings + NeurekaV2RQSDWConv2DTilingReadyBindings)
NeurekaV2DWConv2DMapper = NodeMapper(NeurekaDWConv2DParser(),
                                     NeurekaV2WmemDWConv2DTilingReadyBindings + NeurekaV2DWConv2DTilingReadyBindings)

NeurekaV2RqntDenseConv2DMapper = NodeMapper(
    NeurekaRQSDenseConv2DParser(),
    NeurekaV2WmemRQSDenseConv2DTilingReadyBindings + NeurekaV2RQSDenseConv2DTilingReadyBindings)
NeurekaV2DenseConv2DMapper = NodeMapper(
    NeurekaDenseConv2DParser(), NeurekaV2WmemDenseConv2DTilingReadyBindings + NeurekaV2DenseConv2DTilingReadyBindings)

NeurekaV2Mapping = {
    'RequantizedConv':
        PULPRQSConvLayer([NeurekaV2RqntPWConv2DMapper, NeurekaV2RqntDWConv2DMapper, NeurekaV2RqntDenseConv2DMapper]),
    'Conv':
        ConvLayer([NeurekaV2PWConv2DMapper, NeurekaV2DWConv2DMapper, NeurekaV2DenseConv2DMapper]),
}

_includeList = [
    "pulp_nnx_neureka_v2.h", "pulp_nnx_util.h", "neureka_v2_siracusa_bsp.h", "neureka_v2.h", "neureka_v2_task.h"
]

_neurekaInitCode = r"""
neureka_v2_siracusa_conf_t conf = {.max_stall = 8};
neureka_v2_nnx_init(neureka_v2_siracusa_get_dev(), &conf);
"""


class NeurekaV2Engine(DeploymentEngine):

    def __init__(self,
                 name: str,
                 Mapping = NeurekaV2Mapping,
                 initCode: str = _neurekaInitCode,
                 includeList: List[str] = _includeList,
                 enable3x3: bool = False,
                 enableStrides: bool = False) -> None:
        super().__init__(name, Mapping, initCode, includeList)

        self.enable3x3 = enable3x3
        self.enableStrides = enableStrides

    def isDenseConv(self, node) -> bool:
        return node.op in ["Conv", "RequantizedConv"] and \
            isinstance(node.inputs[1], gs.Constant) and \
            node.attrs['kernel_shape'] == [3, 3] and \
            node.attrs['dilations'] == [1, 1] and \
            node.attrs['group'] == 1 and \
            (node.attrs['strides'] == [1, 1] or self.enableStrides)

    def isPWConv(self, node) -> bool:
        return node.op in ["Conv", "RequantizedConv"] and \
            isinstance(node.inputs[1], gs.Constant) and \
            node.attrs['kernel_shape'] == [1, 1] and \
            node.attrs['dilations'] == [1, 1] and \
            (node.attrs['strides'] == [1, 1] or self.enableStrides)

    def isDWConv(self, node) -> bool:
        return node.op in ["Conv", "RequantizedConv"] and \
            isinstance(node.inputs[1], gs.Constant) and \
            node.attrs['kernel_shape'] == [3, 3] and \
            node.attrs['dilations'] == [1, 1] and \
            node.attrs['group'] != 1 and \
            (node.attrs['strides'] == [1, 1] or self.enableStrides)

    def canExecute(self, node: gs.Node) -> bool:
        if self.enable3x3:
            return self.isPWConv(node) or self.isDWConv(node) or self.isDenseConv(node)
        else:
            return self.isPWConv(node)
