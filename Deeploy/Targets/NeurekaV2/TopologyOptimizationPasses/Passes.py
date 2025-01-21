# ----------------------------------------------------------------------
#
# File: Passes.py
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

from functools import partial

import numpy as np
import numpy.typing as npt
import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.Matchers import Match, NonBranchingMatcher
from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import ReplaceSequentialPatternPass, SequentialPass, \
    contextagnostic
from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import \
    RemoveGlobalOutputReshapePass
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import ReshapeConstOptPass, ReshapeMergePass
from Deeploy.Targets.Neureka.TopologyOptimizationPasses.Passes import NeurekaReshapePointwiseConvolutionPass

_WEIGHT_BANDWIDTH = 288
_CIN_SUBTILE = 32


def _weightEncode(weight: npt.NDArray[np.uint8], bits: int, depthwise: bool = False) -> npt.NDArray[np.uint8]:
    """Unroll weight into expected memory format

    Expected weight shape is (cout, cin, H, W).
    The produced memory layout depends on the weight kernel shape:
      - 3x3: (cout, cinMajor, Bits, H x W x cinMinor_3x3 packed into Weight Bandwidth bits),
      - 1x1: (cout, cinMajor, Bits x H x W x cinMinor_1x1 packed into Weight Bandwidth bits),
    where cinMajor is the ceil(cin / cin subtile <mode>) and cinMinor has to be padded with 0 to cin subtile <mode>.
    """
    if depthwise:
        weight = weight.transpose(1, 0, 2, 3)  # Swap cout and cin

    cout, cin, height, width = weight.shape
    cinSubtile = _CIN_SUBTILE

    # Pad cin to be divisible with CIN_SUBTILE
    if cin % cinSubtile != 0:
        cinPad = cinSubtile - cin % cinSubtile
        weight = np.pad(
            weight,
            ((0, 0), (0, cinPad), (0, 0), (0, 0)),
            "constant",
            constant_values = 0,
        )

    # Reshape into (cout, cinMajor, cinMinor, Flattened spatial, 1)
    # The 1 at the end is required by the unpacking
    cinMajor = int(np.ceil(cin / cinSubtile))
    weight = weight.reshape(cout, cinMajor, cinSubtile, height * width, 1)

    # Unpack 'bits' bits in little order, e.g. bits=4: 3 => [1, 1, 0, 0]
    # (cout, cinMajor, cinSubtile, Flattened spatial, Bits)
    weight = np.unpackbits(weight, axis = -1, count = bits, bitorder = "little")

    # Shuffle bits so that the final shape is:
    # (cout, cinMajor, Bits, Flattened spatial, cinSubtile)
    weight = weight.transpose(0, 1, 4, 3, 2)

    # Pack bits into bytes
    # (-1, 8)
    weight = weight.reshape(-1, 8)
    # (-1, 1)
    weight = np.packbits(weight, axis = -1, bitorder = "little")

    if height == 1 and width == 1:
        # (cout, cinMajor, cinSubtile)
        return weight.reshape(cout, cinMajor, cinSubtile)
    else:
        # (cout, cinMajor, Bits, Weight Bandwidth Bytes)
        return weight.reshape(cout, cinMajor, bits, _WEIGHT_BANDWIDTH // 8)


def _neureka_adjust_weight_memory_layout_fun(graph: gs.Graph, match: Match, name: str, default_channels_first: bool,
                                             engineName: str):
    matched_nodes = list(match.nodes_map.values())
    node = matched_nodes[0]

    if not ("engine" in node.attrs and node.attrs["engine"] == engineName):
        return graph

    weightTensor = node.inputs[1]

    if not isinstance(weightTensor, gs.Constant):
        return graph

    # Adjust N-EUREKA's weights
    values = weightTensor.values

    # Extract weight offset and translate weights by the offset
    weight_offset = values.min()
    values = values - weight_offset
    node.attrs["weight_offset"] = weight_offset

    if "channels_first" in node.attrs:
        channels_first = node.attrs["channels_first"]
    else:
        channels_first = default_channels_first

    # Weight encode expects channels first
    if not channels_first:
        values = values.transpose(0, 3, 1, 2)

    weightBits = 8  # Support only 8 bit weights for now
    weightTensor.values = _weightEncode(values.astype(np.uint8), weightBits, depthwise = node.attrs['group'] == 1)

    weightTensor.name = f"{name}_{weightTensor.name}"

    return graph


@contextagnostic
class NeurekaV2AdjustWeightMemoryLayoutPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool, engineName: str):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['out'], op = 'RequantizedConv|Conv', name = 'node')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        super().__init__(
            graph,
            partial(_neureka_adjust_weight_memory_layout_fun,
                    default_channels_first = default_channels_first,
                    engineName = engineName), "_NEUREKA_ADJUST_WEIGHT_MEMORY_LAYOUT_PASS",
            NonBranchingMatcher(regex_op = True))


@contextagnostic
class NeurekaV2OptimizationPass(SequentialPass):

    def __init__(self, default_channels_first: bool, engineName: str):
        super().__init__(NeurekaV2AdjustWeightMemoryLayoutPass(default_channels_first, engineName),
                         NeurekaReshapePointwiseConvolutionPass(default_channels_first, engineName),
                         ReshapeMergePass(),
                         ReshapeConstOptPass(),
                         RemoveGlobalOutputReshapePass(),
                         name_prefix = '')
