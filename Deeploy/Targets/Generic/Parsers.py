# ----------------------------------------------------------------------
#
# File: BasicParsers.py
#
# Last edited: 15.12.2021
#
# Copyright (C) 2021, ETH Zurich and University of Bologna.
#
# Authors:
# - Moritz Scherer, ETH Zurich
# - Victor Jung, ETH Zurich
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

import math
from typing import Literal, Tuple

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import NetworkContext, NodeParser


class ConcatParser(NodeParser):

    def __init__(self):
        super().__init__(output_sym_names = ['data_out'], required_attrs = ['axis'])

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False
        return len(node.inputs) > 1

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, wellFormed = super().parseNodeCtxt(ctxt, node, channels_first)
        if not wellFormed:
            return ctxt, False

        for idx, tensor in enumerate(node.inputs):
            self.operatorRepresentation[f'data_in_{idx+1}'] = newCtxt.lookup(tensor.name).name
        return newCtxt, True


class iRMSNormParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['data_in', 'weight'],
                         output_sym_names = ['data_out'],
                         required_attrs = ['D', 'n_levels'])

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        self.operatorRepresentation['n_levels'] = int(self.operatorRepresentation['n_levels'])
        self.operatorRepresentation['log2D'] = int(math.log2(self.operatorRepresentation['D']))
        return True


class RQSParserInterface():

    def __init__(self, order: Literal["mul add", "add mul"]) -> None:
        if order == "mul add":
            input_sym_names = ['mul', 'add']
        elif order == "add mul":
            input_sym_names = ['add', 'mul']
        else:
            raise RuntimeError(f"Unrecognized order {order}")

        if self.input_sym_names is not None:
            self.input_sym_names.extend(input_sym_names)
        else:
            self.input_sym_names = input_sym_names

    def parseNode(self, node: gs.Node) -> bool:
        if not all([
                'div' in node.attrs,
                'n_levels' in node.attrs or 'n_levels_out' in node.attrs,
                'signed' in node.attrs,
        ]):
            return False

        n_levels = node.attrs['n_levels'] if 'n_levels' in node.attrs else node.attrs['n_levels_out']
        self.operatorRepresentation['n_levels'] = int(NodeParser._unpack_const(n_levels))
        self.operatorRepresentation['signed'] = int(NodeParser._unpack_const(node.attrs['signed']))
        self.operatorRepresentation['log2D'] = int(math.log2(NodeParser._unpack_const(node.attrs['div'])))
        return True


class SliceParser(NodeParser):

    def __init__(self):
        super().__init__(output_sym_names = ['data_out'])

    def parseNode(self, node: gs.Node) -> bool:
        # Scheremo ONNX >= 10
        retNew = all([len(node.inputs) >= 3, len(node.inputs) <= 5])

        # Scheremo ONNX < 10
        retOld = all([len(node.inputs) == 1, 'ends' in node.attrs, 'starts' in node.attrs])

        return retNew or retOld

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        data_in = ctxt.lookup(node.inputs[0].name)

        self.operatorRepresentation['data_in_shape'] = data_in.shape
        self.operatorRepresentation['dims'] = len(data_in.shape)
        self.operatorRepresentation['data_in'] = data_in.name

        if len(node.inputs) <= 1:
            values = node.attrs['starts']
            startsTensor = gs.Constant(f'{node.name}_Starts_Tensor', values = values)
            ctxt.hoistConstant(startsTensor)
            node.inputs.append(startsTensor)
        if len(node.inputs) <= 2:
            values = node.attrs['ends']
            endsTensor = gs.Constant(f'{node.name}_Ends_Tensor', values = values)
            ctxt.hoistConstant(endsTensor)
            node.inputs.append(endsTensor)
        if len(node.inputs) <= 3:
            values = np.array(list(range(self.operatorRepresentation['dims'])))
            axesTensor = gs.Constant(f'{node.name}_Axes_Tensor', values = values)
            ctxt.hoistConstant(axesTensor)
            node.inputs.append(axesTensor)
        if len(node.inputs) <= 4:
            values = np.ones((self.operatorRepresentation['dims']))
            stepsTensor = gs.Constant(f'{node.name}_Steps_Tensor', values = values)
            ctxt.hoistConstant(stepsTensor)
            node.inputs.append(stepsTensor)

        self.operatorRepresentation['starts'] = node.inputs[1].name
        self.operatorRepresentation['ends'] = node.inputs[2].name

        self.operatorRepresentation['axes'] = node.inputs[3].name
        self.operatorRepresentation['steps'] = node.inputs[4].name

        return ctxt, True


class TransposeParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['data_in'], output_sym_names = ['data_out'], required_attrs = ['perm'])


class MaxPoolParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['data_in'],
                         output_sym_names = ['data_out'],
                         required_attrs = ['ceil_mode', 'kernel_shape', 'pads', 'strides'])


class MaxPool2DParser(MaxPoolParser):

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        if not all([
                len(self.operatorRepresentation['pads']) == 4,
                len(self.operatorRepresentation['kernel_shape']) == 2,
                len(self.operatorRepresentation['strides']) == 2,
        ]):
            return False

        self.operatorRepresentation['padding_x'] = int(self.operatorRepresentation['pads'][0])
        self.operatorRepresentation['padding_y'] = int(self.operatorRepresentation['pads'][1])
        self.operatorRepresentation['padding_x_left'] = int(self.operatorRepresentation['pads'][0])
        self.operatorRepresentation['padding_y_top'] = int(self.operatorRepresentation['pads'][1])
        self.operatorRepresentation['padding_x_right'] = int(self.operatorRepresentation['pads'][2])
        self.operatorRepresentation['padding_y_bottom'] = int(self.operatorRepresentation['pads'][3])
        self.operatorRepresentation['stride_x'] = int(self.operatorRepresentation['strides'][0])
        self.operatorRepresentation['stride_y'] = int(self.operatorRepresentation['strides'][1])
        self.operatorRepresentation['dim_kernel_x'] = int(self.operatorRepresentation['kernel_shape'][0])
        self.operatorRepresentation['dim_kernel_y'] = int(self.operatorRepresentation['kernel_shape'][1])
        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, wellFormed = super().parseNodeCtxt(ctxt, node, channels_first)
        if not wellFormed:
            return ctxt, False

        shape_in = self.operatorRepresentation['data_in_shape']
        shape_out = self.operatorRepresentation['data_out_shape']

        if not all([
                len(shape_in) == 4,
                len(shape_out) == 4,
                shape_in[0] == shape_out[0],
        ]):
            return ctxt, False

        self.operatorRepresentation['batch'] = shape_in[0]
        if channels_first:
            self.operatorRepresentation['ch_im_in'] = shape_in[1]
            self.operatorRepresentation['dim_im_in_x'] = shape_in[2]
            self.operatorRepresentation['dim_im_in_y'] = shape_in[3]
            self.operatorRepresentation['ch_im_out'] = shape_out[1]
            self.operatorRepresentation['dim_im_out_x'] = shape_out[2]
            self.operatorRepresentation['dim_im_out_y'] = shape_out[3]
        else:
            self.operatorRepresentation['ch_im_in'] = shape_in[3]
            self.operatorRepresentation['dim_im_in_x'] = shape_in[1]
            self.operatorRepresentation['dim_im_in_y'] = shape_in[2]
            self.operatorRepresentation['ch_im_out'] = shape_out[3]
            self.operatorRepresentation['dim_im_out_x'] = shape_out[1]
            self.operatorRepresentation['dim_im_out_y'] = shape_out[2]
        return newCtxt, True


class PadParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['data_in'],
                         output_sym_names = ['data_out'],
                         required_attrs = ['mode', 'pads', 'value'])


class Pad2DParser(PadParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        pads = self.operatorRepresentation['pads']
        if len(pads) != 8 or not all([
                pads[0] == 0,  # Batch pad before
                pads[1] == 0,  # Channel pad after
                pads[4] == 0,  # Batch pad before
                pads[5] == 0,  # Channel pad after
        ]):
            return False

        self.operatorRepresentation['pad_x'] = int(pads[3])
        self.operatorRepresentation['pad_y'] = int(pads[2])
        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, wellFormed = super().parseNodeCtxt(ctxt, node, channels_first)
        if not wellFormed:
            return ctxt, False

        shape_in = self.operatorRepresentation['data_in_shape']
        shape_out = self.operatorRepresentation['data_out_shape']

        if not all([
                len(shape_in) == 4,
                len(shape_out) == 4,
                shape_in[0] == shape_out[0],
        ]):
            return ctxt, False

        self.operatorRepresentation['batch'] = shape_in[0]
        if channels_first:
            self.operatorRepresentation['dim_im_in_x'] = shape_in[2]
            self.operatorRepresentation['dim_im_in_y'] = shape_in[3]
            self.operatorRepresentation['dim_im_in_ch'] = shape_in[1]
            self.operatorRepresentation['dim_im_out_x'] = shape_out[2]
            self.operatorRepresentation['dim_im_out_y'] = shape_out[3]
            self.operatorRepresentation['dim_im_out_ch'] = shape_out[1]
        else:
            self.operatorRepresentation['dim_im_in_x'] = shape_in[1]
            self.operatorRepresentation['dim_im_in_y'] = shape_in[2]
            self.operatorRepresentation['dim_im_in_ch'] = shape_in[3]
            self.operatorRepresentation['dim_im_out_x'] = shape_out[1]
            self.operatorRepresentation['dim_im_out_y'] = shape_out[2]
            self.operatorRepresentation['dim_im_out_ch'] = shape_out[3]
        return newCtxt, True


class Pad1DParser(PadParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        pads = self.operatorRepresentation['pads']
        if len(pads) != 6 or not all([
                pads[0] == 0,  # Batch pad before
                pads[1] == 0,  # Channel pad before
                pads[3] == 0,  # Batch pad after
                pads[4] == 0,  # Channel pad after
        ]):
            return False

        self.operatorRepresentation['pad_x'] = 0
        self.operatorRepresentation['pad_y'] = int(pads[2])
        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, wellFormed = super().parseNodeCtxt(ctxt, node, channels_first)
        if not wellFormed:
            return ctxt, False

        shape_in = self.operatorRepresentation['data_in_shape']
        shape_out = self.operatorRepresentation['data_out_shape']

        if not all([
                len(shape_in) == 3,
                len(shape_out) == 3,
                shape_in[0] == shape_out[0],
        ]):
            return ctxt, False

        self.operatorRepresentation['batch'] = shape_in[0]
        self.operatorRepresentation['dim_im_in_x'] = 1
        self.operatorRepresentation['dim_im_out_x'] = 1
        if channels_first:
            self.operatorRepresentation['dim_im_in_y'] = shape_in[2]
            self.operatorRepresentation['dim_im_in_ch'] = shape_in[1]
            self.operatorRepresentation['dim_im_out_y'] = shape_out[2]
            self.operatorRepresentation['dim_im_out_ch'] = shape_out[1]
        else:
            self.operatorRepresentation['dim_im_in_y'] = shape_in[1]
            self.operatorRepresentation['dim_im_in_ch'] = shape_in[2]
            self.operatorRepresentation['dim_im_out_y'] = shape_out[1]
            self.operatorRepresentation['dim_im_out_ch'] = shape_out[2]
        return newCtxt, True


class AddParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['data_in_1', 'data_in_2'], output_sym_names = ['data_out'])


class ReduceParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['data_in'],
                         output_sym_names = ['data_out'],
                         required_attrs = ['axes', 'keepdims'])

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        if isinstance(self.operatorRepresentation['axes'], int):
            self.operatorRepresentation['axes'] = [self.operatorRepresentation['axes']]
        self.operatorRepresentation['keepdims'] = int(self.operatorRepresentation['keepdims'])
        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, wellFormed = super().parseNodeCtxt(ctxt, node, channels_first)
        if not wellFormed:
            return ctxt, False

        shape_in = self.operatorRepresentation['data_in_shape']
        first_axis = self.operatorRepresentation['axes'][0]
        self.operatorRepresentation['axisLength'] = shape_in[first_axis]
        return newCtxt, True


class ReduceMeanParser(ReduceParser):
    pass


class ReduceSumParser(ReduceParser):
    pass


class SoftmaxParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['data_in'], output_sym_names = ['data_out'], optional_attrs = {'axis': -1})

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        self.operatorRepresentation['axis'] = int(self.operatorRepresentation['axis'])
        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, wellFormed = super().parseNodeCtxt(ctxt, node, channels_first)
        if not wellFormed:
            return ctxt, False

        shape_in = self.operatorRepresentation['data_in_shape']
        axis = self.operatorRepresentation['axis']
        self.operatorRepresentation['lastDimLength'] = shape_in[axis]
        return newCtxt, True


class SoftmaxGradParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['upstream_grad', 'softmax_output'],
                         output_sym_names = ['softmax_grad'],
                         optional_attrs = {'axis': -1})

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        self.operatorRepresentation['axis'] = int(self.operatorRepresentation['axis'])
        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, wellFormed = super().parseNodeCtxt(ctxt, node, channels_first)
        if not wellFormed:
            return ctxt, False

        upstream_grad_shape = self.operatorRepresentation['upstream_grad_shape']
        axis = self.operatorRepresentation['axis']
        self.operatorRepresentation['lastDimLength'] = upstream_grad_shape[axis]
        return ctxt, True


class iSoftmaxParser(SoftmaxParser):

    def __init__(self):
        super().__init__()
        required_attrs = ['coeffA', 'coeffB', 'coeffC', 'log2', 'n_levels']
        if self.required_attrs is not None:
            self.required_attrs.extend(required_attrs)
        else:
            self.required_attrs = required_attrs

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        self.operatorRepresentation['coeffA'] = int(self._unpack_const(self.operatorRepresentation['coeffA']))
        self.operatorRepresentation['coeffB'] = int(self._unpack_const(self.operatorRepresentation['coeffB']))
        self.operatorRepresentation['coeffC'] = int(self._unpack_const(self.operatorRepresentation['coeffC']))
        self.operatorRepresentation['log2'] = int(self._unpack_const(self.operatorRepresentation['log2']))
        self.operatorRepresentation['n_levels'] = int(self._unpack_const(self.operatorRepresentation['n_levels']))
        return True


class ITAMaxParser(SoftmaxParser):

    def __init__(self):
        super().__init__()
        required_attrs = ['n_levels']
        if self.required_attrs is not None:
            self.required_attrs.extend(required_attrs)
        else:
            self.required_attrs = required_attrs

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        self.operatorRepresentation['n_levels'] = int(self._unpack_const(self.operatorRepresentation['n_levels']))
        return True


class ITAPartialMaxParser(ITAMaxParser):

    def __init__(self):
        super().__init__()
        required_attrs = ['group_width']
        if self.required_attrs is not None:
            self.required_attrs.extend(required_attrs)
        else:
            self.required_attrs = required_attrs

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        self.operatorRepresentation['group_width'] = int(self._unpack_const(self.operatorRepresentation['group_width']))
        return True


class GELUParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['data_in'], output_sym_names = ['data_out'], required_attrs = ['b', 'one'])


class RQSiGELUParser(GELUParser):

    def __init__(self):
        super().__init__()
        input_sym_names = ['mul', 'add', 'shift']
        if self.input_sym_names is not None:
            self.input_sym_names.extend(input_sym_names)
        else:
            self.input_sym_names = input_sym_names


class iHardswishParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['data_in'],
                         output_sym_names = ['data_out'],
                         required_attrs = ['one_over_six', 'six', 'three'])


class iNoNormParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['data_in', 'weights', 'bias'],
                         output_sym_names = ['data_out'],
                         required_attrs = ['D', 'mul', 'n_levels'])

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        self.operatorRepresentation['log2D'] = int(np.log2(self._unpack_const(self.operatorRepresentation['D'])))
        self.operatorRepresentation['mul'] = int(self._unpack_const(self.operatorRepresentation['mul']))
        return True


class RQSiHardswishParser(iHardswishParser):

    def __init__(self):
        super().__init__()
        required_attrs = ['mul', 'add', 'shift']
        if self.required_attrs is not None:
            self.required_attrs.extend(required_attrs)
        else:
            self.required_attrs = required_attrs


class GatherParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['data_in', 'indices'],
                         output_sym_names = ['data_out'],
                         optional_attrs = {'axis': 0})

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, wellFormed = super().parseNodeCtxt(ctxt, node, channels_first)
        if not wellFormed:
            return ctxt, False

        indices_shape = self.operatorRepresentation['indices_shape']

        # Only indices of size 1 supported
        if not np.prod(indices_shape) == 1:
            return ctxt, False
        self.operatorRepresentation['index'] = int(self._unpack_const(node.inputs[1]))

        axis = self.operatorRepresentation['axis']
        shape = self.operatorRepresentation['data_in_shape']
        self.operatorRepresentation['batch'] = np.prod(shape[:axis])
        self.operatorRepresentation['batch_length'] = np.prod(shape[axis:])
        self.operatorRepresentation['axis_length'] = np.prod(shape[axis + 1:])
        return newCtxt, True


class FlattenParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['data_in'], output_sym_names = ['data_out'], required_attrs = ['axis'])


class UnsqueezeParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['data_in'], output_sym_names = ['data_out'], required_attrs = ['axes'])


class ReluParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['data_in'], output_sym_names = ['data_out'])


class ReshapeParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['data_in', 'shape'], output_sym_names = ['data_out'])


class RequantShiftParser(NodeParser, RQSParserInterface):

    def __init__(self):
        NodeParser.__init__(self)
        RQSParserInterface.__init__(self, "mul add")

    def parseNode(self, node: gs.Node) -> bool:
        return NodeParser.parseNode(self, node) and RQSParserInterface.parseNode(self, node)

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, wellFormed = super().parseNodeCtxt(ctxt, node, channels_first)
        if not wellFormed:
            return ctxt, False

        shape = self.operatorRepresentation['data_in_shape']

        # Supported shape lengths greater then 2
        if not len(shape) >= 2:
            return ctxt, False

        # Assumes shape [ Batch, Channels, ...]
        self.operatorRepresentation['batch'] = shape[0]
        self.operatorRepresentation['channels'] = shape[1]
        self.operatorRepresentation['channel_width'] = np.prod(shape[2:]) if len(shape) > 2 else 1

        return ctxt, True


class UniformRequantShiftParser(RequantShiftParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        return all([
            np.prod(node.inputs[1].values.shape) == 1,
            np.prod(node.inputs[2].values.shape) == 1,
        ])


class MulParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['A', 'B'], output_sym_names = ['C'])


class ConvParser(NodeParser):

    def __init__(self, noBiasHoisting):
        input_sym_names = ['data_in', 'weight']
        if not noBiasHoisting:
            input_sym_names.append('bias')
        super().__init__(input_sym_names = input_sym_names,
                         output_sym_names = ['data_out'],
                         required_attrs = ['dilations', 'group', 'pads', 'strides'])


class Conv2DParser(ConvParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)
        optional_attrs = {'bias_shift': 0, 'out_shift': 0}
        if self.optional_attrs is not None:
            self.optional_attrs.update(optional_attrs)
        else:
            self.optional_attrs = optional_attrs

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        if not all([
                # Make sure strides are 2D
                len(self.operatorRepresentation['strides']) == 2,
                len(self.operatorRepresentation['pads']) == 4,
                len(self.operatorRepresentation['dilations']) == 2,
        ]):
            return False

        if 'kernel_shape' not in node.attrs:
            node.attrs['kernel_shape'] = node.inputs[1].shape[-2:]
        self.operatorRepresentation['kernel_shape'] = node.attrs['kernel_shape']
        self.operatorRepresentation['dim_kernel_x'] = int(self.operatorRepresentation['kernel_shape'][0])
        self.operatorRepresentation['dim_kernel_y'] = int(self.operatorRepresentation['kernel_shape'][1])
        self.operatorRepresentation['dilation_x'] = int(self.operatorRepresentation['dilations'][0])
        self.operatorRepresentation['dilation_y'] = int(self.operatorRepresentation['dilations'][1])
        self.operatorRepresentation['padding_x'] = int(self.operatorRepresentation['pads'][0])
        self.operatorRepresentation['padding_y'] = int(self.operatorRepresentation['pads'][1])
        self.operatorRepresentation['stride_x'] = int(self.operatorRepresentation['strides'][0])
        self.operatorRepresentation['stride_y'] = int(self.operatorRepresentation['strides'][1])
        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, wellFormed = super().parseNodeCtxt(ctxt, node, channels_first)
        if not wellFormed:
            return ctxt, False

        shape_in = self.operatorRepresentation('data_in_shape')
        shape_weight = self.operatorRepresentation('weight_shape')
        shape_out = self.operatorRepresentation('data_out_shape')

        if not all([
                len(shape_in) == 4,
                len(shape_weight) == 4,
                len(shape_out) == 4,
                shape_in[0] == shape_out[0],
        ]):
            return ctxt, False

        self.operatorRepresentation['batch'] = shape_in[0]
        if channels_first:
            self.operatorRepresentation['ch_im_in'] = shape_in[1]
            self.operatorRepresentation['dim_im_in_x'] = shape_in[2]
            self.operatorRepresentation['dim_im_in_y'] = shape_in[3]
            self.operatorRepresentation['ch_im_out'] = shape_out[1]
            self.operatorRepresentation['dim_im_out_x'] = shape_out[2]
            self.operatorRepresentation['dim_im_out_y'] = shape_out[3]
        else:
            self.operatorRepresentation['ch_im_in'] = shape_in[3]
            self.operatorRepresentation['dim_im_in_x'] = shape_in[1]
            self.operatorRepresentation['dim_im_in_y'] = shape_in[2]
            self.operatorRepresentation['ch_im_out'] = shape_out[3]
            self.operatorRepresentation['dim_im_out_x'] = shape_out[1]
            self.operatorRepresentation['dim_im_out_y'] = shape_out[2]
        return newCtxt, True


class RQSConv2DParser(Conv2DParser, RQSParserInterface):

    def __init__(self):
        # LMACAN: We don't hoist the bias because the bias is incorporated inside that `add`
        # tensor of requantization.
        Conv2DParser.__init__(self, noBiasHoisting = True)
        RQSParserInterface.__init__(self, "mul add")

    def parseNode(self, node: gs.Node) -> bool:
        return RQSParserInterface.parseNode(self, node) and Conv2DParser.parseNode(self, node)


class Conv1DParser(ConvParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__(noBiasHoisting)

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        if not all([
                # Make sure strides are 1D
                len(self.operatorRepresentation['strides']) == 1,
                len(self.operatorRepresentation['pads']) == 2,
                len(self.operatorRepresentation['dilations']) == 1,
        ]):
            return False

        if 'kernel_shape' not in node.attrs:
            node.attrs['kernel_shape'] = node.inputs[1].shape[-1:]
        self.operatorRepresentation['kernel_shape'] = node.attrs['kernel_shape']
        self.operatorRepresentation['dim_kernel_y'] = int(self.operatorRepresentation['kernel_shape'][0])
        self.operatorRepresentation['dilation_y'] = int(self.operatorRepresentation['dilations'][0])
        self.operatorRepresentation['padding_y'] = int(self.operatorRepresentation['pads'][0])
        self.operatorRepresentation['stride_y'] = int(self.operatorRepresentation['strides'][0])
        self.operatorRepresentation['bias_shift'] = int(0)
        self.operatorRepresentation['out_shift'] = int(0)
        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)
        if not ret:
            return ctxt, False

        shape_in = self.operatorRepresentation('data_in_shape')
        shape_weight = self.operatorRepresentation('weight_shape')
        shape_out = self.operatorRepresentation('data_out_shape')

        if not all([
                len(shape_in) == 3,
                len(shape_weight) == 3,
                len(shape_out) == 3,
                shape_in[0] == shape_out[0],
        ]):
            return ctxt, False

        self.operatorRepresentation['batch'] = shape_in[0]
        self.operatorRepresentation['dim_im_in_x'] = 1
        self.operatorRepresentation['dim_im_out_x'] = 1

        if channels_first:
            self.operatorRepresentation['ch_im_in'] = shape_in[1]
            self.operatorRepresentation['dim_im_in_y'] = shape_in[2]
            self.operatorRepresentation['ch_im_out'] = shape_out[1]
            self.operatorRepresentation['dim_im_out_y'] = shape_out[2]
        else:
            self.operatorRepresentation['ch_im_in'] = shape_in[2]
            self.operatorRepresentation['dim_im_in_y'] = shape_in[1]
            self.operatorRepresentation['ch_im_out'] = shape_out[2]
            self.operatorRepresentation['dim_im_out_y'] = shape_out[1]
        return newCtxt, True


class RQSConv1DParser(Conv1DParser, RQSParserInterface):

    def __init__(self):
        Conv1DParser.__init__(self, noBiasHoisting = True)
        RQSParserInterface.__init__(self, "mul add")

    def parseNode(self, node: gs.Node) -> bool:
        return RQSParserInterface.parseNode(self, node) and Conv1DParser.parseNode(self, node)


class MHSAParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = [
            'q', 'k', 'v', 'wq_weight', 'wq_bias', 'wk_weight', 'wk_bias', 'wv_weight', 'wv_bias', 'wo_weight',
            'wo_bias'
        ],
                         output_sym_names = ['data_out'],
                         required_attrs = [
                             'preattn_requant_mul', 'preattn_requant_div', 'postattn_requant_mul',
                             'postattn_requant_div', 'wo_requant_mul', 'wo_requant_div', 'wq_requant_mul',
                             'wq_requant_div', 'wk_requant_mul', 'wk_requant_div', 'wv_requant_mul', 'wv_requant_div',
                             'n_levels', 'dim', 'dim_head', 'heads', 'signed'
                         ])

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        self.operatorRepresentation['n_levels'] = int(self._unpack_const(self.operatorRepresentation['n_levels']))
        self.operatorRepresentation['dim'] = int(self._unpack_const(
            self.operatorRepresentation['dim']))  # Sequence Length
        self.operatorRepresentation['dim_head'] = int(self._unpack_const(
            self.operatorRepresentation['dim_head']))  # Projection Size
        self.operatorRepresentation['heads'] = int(self._unpack_const(self.operatorRepresentation['heads']))
        self.operatorRepresentation['signed'] = int(self._unpack_const(self.operatorRepresentation['signed']))
        return True


class LinearAttentionParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = [
            'q', 'k', 'v', 'wq_weight', 'wq_bias', 'wk_weight', 'wk_bias', 'wv_weight', 'wv_bias', 'wo_weight',
            'wo_bias'
        ],
                         output_sym_names = ['data_out'],
                         required_attrs = [
                             'preattn_requant_mul', 'preattn_requant_div', 'normalizer_requant_mul',
                             'normalizer_requant_div', 'postattn_requant_mul', 'postattn_requant_div', 'wo_requant_mul',
                             'wo_requant_div', 'wq_requant_mul', 'wq_requant_div', 'wk_requant_mul', 'wk_requant_div',
                             'wv_requant_mul', 'wv_requant_div', 'Delta', 'eps', 'act_type', 'n_levels', 'dim',
                             'dim_head', 'heads'
                         ])

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        def int_unpack(attr: str) -> int:
            return int(self._unpack_const(self.operatorRepresentation[attr]))

        self.operatorRepresentation.update({
            attr: int_unpack(attr) for attr in [
                'preattn_requant_mul',
                'preattn_requant_shift',
                'normalizer_requant_mul',
                'normalizer_requant_shift',
                'postattn_requant_mul',
                'postattn_requant_shift',
                'wo_requant_mul',
                'wo_requant_shift',
                'wq_requant_mul',
                'wq_requant_shift',
                'wk_requant_mul',
                'wk_requant_shift',
                'wv_requant_mul',
                'wv_requant_shift',
                'Delta',
                'eps',
                'act_type',
                'n_levels',
                'dim',
                'dim_head',
                'heads',
            ]
        })

        def log2_int_unpack(attr: str) -> int:
            return int(math.log2(self._unpack_const(self.operatorRepresentation[attr])))

        self.operatorRepresentation.update({
            attr: log2_int_unpack(attr) for attr in [
                'preattn_requant_div',
                'normalizer_requant_div',
                'postattn_requant_div',
                'wo_requant_div',
                'wq_requant_div',
                'wk_requant_div',
                'wv_requant_div',
            ]
        })

        return True


class CLCAParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = [
            'q', 'k', 'wq_weight', 'wq_bias', 'wk_weight', 'wk_bias', 'wo_weight', 'wo_bias', 'wq_requant_mul',
            'wq_requant_add', 'wq_requant_div', 'wk_requant_mul', 'wk_requant_add', 'wk_requant_div', 'wv_requant_mul',
            'wv_requant_add', 'wv_requant_div', 'kdiv_requant_mul', 'kdiv_requant_add', 'kdiv_requant_div',
            'preattn_requant_mul', 'preattn_requant_add', 'preattn_requant_div', 'postattn_requant_mul',
            'postattn_requant_add', 'postattn_requant_div', 'wo_requant_mul', 'wo_requant_add', 'wo_requant_div'
        ],
                         output_sym_names = ['data_out'],
                         required_attrs = [
                             'Delta', 'eps', 'eta', 'act_type', 'n_levels', 'dim', 'dim_head', 'out_dim', 'heads'
                         ])

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        def int_unpack(attr: str) -> int:
            return int(self._unpack_const(self.operatorRepresentation[attr]))

        assert self.required_attrs is not None, "Error: required_attrs is None"
        self.operatorRepresentation.update({attr: int_unpack(attr) for attr in self.required_attrs})
        return True


class iLayerNormParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['data_in', 'weight', 'bias'],
                         output_sym_names = ['data_out'],
                         required_attrs = ['D', 'n_levels'])

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        def int_unpack(attr: str) -> int:
            return int(self._unpack_const(self.operatorRepresentation[attr]))

        self.operatorRepresentation['n_levels'] = int_unpack('n_levels')

        def log2_int_unpack(attr: str) -> int:
            return int(math.log2(self._unpack_const(self.operatorRepresentation[attr])))

        self.operatorRepresentation['log2D'] = log2_int_unpack('D')
        return True


class LayerNormParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['data_in', 'weight', 'bias'],
                         output_sym_names = ['data_out'],
                         required_attrs = ['epsilon'])


class MatMulParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['A', 'B'],
                         output_sym_names = ['data_out'],
                         optional_attrs = {
                             'alpha': 1,
                             'beta': 1,
                             'transB': 0,
                             'transA': 0
                         })

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, wellFormed = super().parseNodeCtxt(ctxt, node, channels_first)
        if not wellFormed:
            return ctxt, False

        shape_A = self.operatorRepresentation['A_shape']
        shape_B = self.operatorRepresentation['B_shape']

        if self.operatorRepresentation['transA'] == 1:
            shape_A_N, shape_A_M = shape_A[-2:]
        else:
            shape_A_M, shape_A_N = shape_A[-2:]

        if self.operatorRepresentation['transB'] == 1:
            shape_B_O, shape_B_N = shape_B[-2:]
        else:
            shape_B_N, shape_B_O = shape_B[-2:]

        shape_A_batch = np.prod(shape_A[:-2])
        shape_B_batch = np.prod(shape_B[:-2])

        if not all([
                shape_A_N == shape_B_N,
                shape_A_batch == shape_B_batch,
        ]):
            return ctxt, False

        self.operatorRepresentation['batch'] = shape_A_batch
        self.operatorRepresentation['M'] = shape_A_M
        self.operatorRepresentation['N'] = shape_A_N
        self.operatorRepresentation['O'] = shape_B_O
        return newCtxt, True


class RQMatMulParser(MatMulParser, RQSParserInterface):

    def __init__(self):
        MatMulParser.__init__(self)
        RQSParserInterface.__init__(self, "add mul")

    def parseNode(self, node: gs.Node) -> bool:
        return MatMulParser.parseNode(self, node) and RQSParserInterface.parseNode(self, node)


# This parser combines Matmul nodes and GEMM nodes to the more general GEMM nodes
class GEMMParser(MatMulParser):

    def __init__(self, noBiasHoisting = True):
        super().__init__()
        assert self.input_sym_names is not None, "Assumes we are using input_sym_names"
        if not noBiasHoisting:
            self.input_sym_names.append('C')

    def parseNode(self, node: gs.Node) -> bool:
        assert self.input_sym_names is not None, "Assumes we are using input_sym_names"
        # We want the bias hosted but there is no bias in the graph
        if len(node.inputs) == 2 and 'C' in self.input_sym_names:
            node.inputs.append(gs.Constant(f'{node.name}_C_Tensor', values = np.zeros((1))))

        return super().parseNode(node)


class RQGEMMParser(GEMMParser, RQSParserInterface):

    def __init__(self, noBiasHoisting = True):
        GEMMParser.__init__(self, noBiasHoisting)
        RQSParserInterface.__init__(self, "add mul")

    def parseNode(self, node: gs.Node) -> bool:
        assert self.input_sym_names is not None, "Assumes we are using input_sym_names"
        # We want the bias hosted but there is no bias in the graph
        if len(node.inputs) == 4 and 'C' in self.input_sym_names:
            node.inputs.append(gs.Constant(f'{node.name}_C_Tensor', values = np.zeros((1))))
        return GEMMParser.parseNode(self, node) and RQSParserInterface.parseNode(self, node)


class DummyParser(NodeParser):

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        inputs = [ctxt.lookup(tensor.name) for tensor in node.inputs]
        outputs = [ctxt.lookup(tensor.name) for tensor in node.outputs]

        self.operatorRepresentation['data_in'] = inputs[0].name
        self.operatorRepresentation['data_out'] = outputs[0].name
        return ctxt, True


class IntegerDivParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['A', 'B'],
                         output_sym_names = ['C'],
                         required_attrs = ['Delta', 'eps', 'eta'])

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, wellFormed = super().parseNodeCtxt(ctxt, node, channels_first)

        shape_A = self.operatorRepresentation['A_shape']
        shape_B = self.operatorRepresentation['B_shape']

        for idx, (a, b) in enumerate(zip(shape_A, shape_B)):
            if a != b:
                self.operatorRepresentation['nomStep'] = np.prod(shape_A[idx:])
                self.operatorRepresentation['denomStep'] = np.prod(shape_B[idx:])
                break

        return ctxt, True


class DivParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['input1', 'input2'], output_sym_names = ['output'])


class RQIntegerDivParser(IntegerDivParser, RQSParserInterface):

    def __init__(self):
        IntegerDivParser.__init__(self)
        RQSParserInterface.__init__(self, "mul add")
        self.input_sym_names = ["A", "B", "requant_mul", "requant_add", "requant_div"]

    def parseNode(self, node: gs.Node) -> bool:
        return IntegerDivParser.parseNode(self, node) and RQSParserInterface.parseNode(self, node)


class DebugParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['data_in'], output_sym_names = ['data_out'])

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, wellFormed = self.parseNodeCtxt(ctxt, node, channels_first)
        if not wellFormed:
            return ctxt, False

        shape_in = self.operatorRepresentation['data_in_shape']
        shape_out = self.operatorRepresentation['data_out_shape']

        if not (len(shape_in) >= 2 and len(shape_in) <= 4):
            return ctxt, False

        # default values
        self.operatorRepresentation['batch'] = shape_in[0]
        self.operatorRepresentation['dim_im_in_x'] = 1
        self.operatorRepresentation['dim_im_in_ch'] = 1
        self.operatorRepresentation['dim_im_out_x'] = 1
        self.operatorRepresentation['dim_im_out_ch'] = 1

        if len(shape_in) == 2:
            self.operatorRepresentation['dim_im_in_y'] = shape_in[1]
            self.operatorRepresentation['dim_im_out_y'] = shape_out[1]
        elif len(shape_in) == 3:
            self.operatorRepresentation['dim_im_in_x'] = shape_in[1]
            self.operatorRepresentation['dim_im_in_y'] = shape_in[2]
            self.operatorRepresentation['dim_im_out_x'] = shape_out[1]
            self.operatorRepresentation['dim_im_out_y'] = shape_out[2]
        elif len(shape_in) == 4:
            if channels_first:
                self.operatorRepresentation['dim_im_in_ch'] = shape_in[1]
                self.operatorRepresentation['dim_im_in_x'] = shape_in[2]
                self.operatorRepresentation['dim_im_in_y'] = shape_in[3]
                self.operatorRepresentation['dim_im_out_ch'] = shape_out[1]
                self.operatorRepresentation['dim_im_out_x'] = shape_out[2]
                self.operatorRepresentation['dim_im_out_y'] = shape_out[3]
            else:
                self.operatorRepresentation['dim_im_in_x'] = shape_in[1]
                self.operatorRepresentation['dim_im_in_y'] = shape_in[2]
                self.operatorRepresentation['dim_im_in_ch'] = shape_in[3]
                self.operatorRepresentation['dim_im_out_x'] = shape_out[1]
                self.operatorRepresentation['dim_im_out_y'] = shape_out[2]
                self.operatorRepresentation['dim_im_out_ch'] = shape_out[3]

        return newCtxt, True


class GenericMaxPool2DParser(MaxPool2DParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        return all([
            all([pad == 0 for pad in self.operatorRepresentation['pads']]),
            self.operatorRepresentation['ceil_mode'] == 0,
        ])


class GenericConv1DParser(Conv1DParser):

    def __init__(self):
        super().__init__(noBiasHoisting = True)

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        return all([
            self.operatorRepresentation['group'] == 1,
            # Make sure padding is square and all 0
            self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][1],
            self.operatorRepresentation['pads'][0] == 0,
            all([coeff == 1 for coeff in self.operatorRepresentation['dilations']]),
        ])


class GenericDWConv1DParser(GenericConv1DParser):

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:

        newCtxt, wellFormed = super().parseNodeCtxt(ctxt, node, channels_first)
        if not wellFormed:
            return ctxt, False

        if self.operatorRepresentation['group'] != self.operatorRepresentation['ch_im_in']:
            return ctxt, False

        return newCtxt, True


class GenericConv2DParser(Conv2DParser):

    def __init__(self):
        super().__init__(noBiasHoisting = True)

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        return all([
            # Make sure padding is square
            self.operatorRepresentation['group'] == 1,
            self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][2],
            self.operatorRepresentation['pads'][1] == self.operatorRepresentation['pads'][3],
            self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][1],
            self.operatorRepresentation['pads'][0] == 0,
            all([coeff == 1 for coeff in self.operatorRepresentation['dilations']]),
        ])


class GenericDWConv2DParser(GenericConv2DParser):

    def __init__(self):
        super().__init__()

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, wellFormed = super().parseNodeCtxt(ctxt, node, channels_first)
        if not wellFormed:
            return ctxt, False

        if self.operatorRepresentation['group'] != self.operatorRepresentation['ch_im_in']:
            return ctxt, False

        return newCtxt, True


class GenericGEMMParser(GEMMParser):

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, wellFormed = super().parseNodeCtxt(ctxt, node, channels_first)
        if not wellFormed:
            return ctxt, False

        # Try to scale A offline if possible, else fail
        if not self.operatorRepresentation['alpha'].is_integer():
            nameA = self.operatorRepresentation['A']
            if newCtxt.is_global(nameA) and isinstance(newCtxt.lookup(nameA), ConstantBuffer):
                A = newCtxt.lookup(nameA)
                npA = np.asarray(A.values).reshape(A.shape)
                newA = npA * self.operatorRepresentation['beta']
                newCtxt.globalObjects[nameA].values = newA
                self.operatorRepresentation['alpha'] = 1.0
            else:
                return ctxt, False
        # Try to scale B offline if possible, else fail
        if not self.operatorRepresentation['beta'].is_integer():
            nameB = self.operatorRepresentation['B']
            if newCtxt.is_global(nameB) and isinstance(newCtxt.lookup(nameB), ConstantBuffer):
                B = newCtxt.lookup(nameB)
                npB = np.asarray(B.values).reshape(B.shape)
                newB = npB * self.operatorRepresentation['beta']
                newCtxt.globalObjects[nameB].values = newB
                self.operatorRepresentation['beta'] = 1.0
            else:
                return ctxt, False

        self.operatorRepresentation['alpha'] = int(self.operatorRepresentation['alpha'])
        self.operatorRepresentation['beta'] = int(self.operatorRepresentation['beta'])
        return newCtxt, True


class RQAddParser(AddParser):

    def __init__(self):
        super().__init__()
        self.required_attrs = [
            'rqs1_mul', 'rqs1_add', 'rqs1_div', 'rqs1_signed', 'rqs2_mul', 'rqs2_add', 'rqs2_div', 'rqs2_signed',
            'rqsOut_mul', 'rqsOut_add', 'rqsOut_div', 'rqsOut_signed'
        ]

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        if not all([
                any(['rqs1_n_levels' in node.attrs, 'rqs1_n_levels_out' in node.attrs]),
                any(['rqs2_n_levels' in node.attrs, 'rqs2_n_levels_out' in node.attrs]),
                any(['rqsOut_n_levels' in node.attrs, 'rqsOut_n_levels_out' in node.attrs]),
        ]):
            return False

        def int_unpack(attr: str) -> int:
            return int(self._unpack_const(self.operatorRepresentation[attr]))

        def log2_int_unpack(attr: str) -> int:
            return int(math.log2(self._unpack_const(self.operatorRepresentation[attr])))

        def parse_rqs(name: str) -> None:
            # Find n_levels and unpack it to integer
            n_levels_attr = f'{name}_n_levels' if f'{name}_n_levels' in node.attrs else f'{name}_n_levels_out'
            self.operatorRepresentation[f'{name}_n_levels'] = int_unpack(n_levels_attr)

            # Unpack mul, add, signed to integer
            self.operatorRepresentation.update(
                {attr: int_unpack(attr) for attr in [
                    f'{name}_mul',
                    f'{name}_add',
                    f'{name}_signed',
                ]})

            # Unpack div to log2D
            self.operatorRepresentation[f'{name}_log2D'] = log2_int_unpack(f'{name}_div')

        parse_rqs('rqs1')
        parse_rqs('rqs2')
        parse_rqs('rqsOut')
        return True


class QuantParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['data_in'],
                         output_sym_names = ['data_out'],
                         required_attrs = ['scale', 'zero_point', 'bit_width'],
                         optional_attrs = {'signed': True})

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        self.operatorRepresentation['scale'] = float(self.operatorRepresentation['scale'])
        self.operatorRepresentation['zero_point'] = float(self.operatorRepresentation['zero_point'])
        self.operatorRepresentation['bit_width'] = int(self.operatorRepresentation['bit_width'])
        self.operatorRepresentation['signed'] = bool(self.operatorRepresentation['signed'])

        # Calculate min and max values based on bit_width and signed
        bit_width_int = self.operatorRepresentation['bit_width']
        if self.operatorRepresentation['signed']:
            self.operatorRepresentation['min_val'] = -(2**(bit_width_int - 1))
            self.operatorRepresentation['max_val'] = (2**(bit_width_int - 1)) - 1
        else:
            self.operatorRepresentation['min_val'] = 0
            self.operatorRepresentation['max_val'] = (2**bit_width_int) - 1
        return True


class DequantParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['data_in'],
                         output_sym_names = ['data_out'],
                         required_attrs = ['scale', 'zero_point', 'bit_width'])

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        self.operatorRepresentation['scale'] = float(self.operatorRepresentation['scale'])
        self.operatorRepresentation['zero_point'] = float(self.operatorRepresentation['zero_point'])
        self.operatorRepresentation['bit_width'] = int(self.operatorRepresentation['bit_width'])
        self.operatorRepresentation['signed'] = bool(self.operatorRepresentation['signed'])
        return True


class SoftmaxCrossEntropyLossParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['logits', 'labels'], output_sym_names = ['log_prob'])

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, wellFormed = super().parseNodeCtxt(ctxt, node, channels_first)
        if not wellFormed:
            return ctxt, False

        logits_shape = self.operatorRepresentation['logits_shape']
        self.operatorRepresentation['batch'] = logits_shape[0]
        self.operatorRepresentation['num_classes'] = logits_shape[1]
        return newCtxt, True


class SoftmaxCrossEntropyLossGradParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['log_prob', 'labels'], output_sym_names = ['grad'])

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, wellFormed = super().parseNodeCtxt(ctxt, node, channels_first)
        if not wellFormed:
            return ctxt, False

        log_prob_shape = self.operatorRepresentation['log_prob_shape']
        self.operatorRepresentation['batch'] = log_prob_shape[0]  # RW: used for tiling
        self.operatorRepresentation['total_batch'] = log_prob_shape[0]  # RW: total batch num for normalization
        self.operatorRepresentation['num_classes'] = log_prob_shape[1]
        return newCtxt, True


class SGDParser(NodeParser):

    def __init__(self):
        super().__init__(input_sym_names = ['weight', 'grad'],
                         output_sym_names = ['weight_update'],
                         required_attrs = ['lr'])
