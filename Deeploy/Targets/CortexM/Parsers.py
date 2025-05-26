# ----------------------------------------------------------------------
#
# File: CMSISParsers.py
#
# Last edited: 17.12.2021
#
# Copyright (C) 2021, ETH Zurich and University of Bologna.
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

from typing import Tuple

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext
from Deeploy.Targets.Generic.Parsers import CLCAParser, GEMMParser, LinearAttentionParser, MaxPool2DParser, \
    MHSAParser, RQSConv1DParser, RQSConv2DParser, RQSParserInterface


class CMSISMaxPool2DParser(MaxPool2DParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        return all([
            self.operatorRepresentation['pads'][0] == 0,
            self.operatorRepresentation['pads'][1] == 0,
        ])


class CMSISDWConv2DParser(RQSConv2DParser):

    def __init__(self):
        super().__init__()
        assert self.input_sym_names is not None, "Assuming input_sym_names are used."
        self.input_sym_names.append('shift')

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        return all(pad == 0 for pad in self.operatorRepresentation['pads'])

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, wellFormed = super().parseNodeCtxt(ctxt, node, channels_first)
        if not wellFormed:
            return ctxt, False

        if not self.operatorRepresentation['group'] == self.operatorRepresentation['weight_shape'][0]:
            return ctxt, False

        if not newCtxt.is_global(self.operatorRepresentation['weight']):
            return ctxt, False

        # SCHEREMO: Transpose weights to be num filters last
        weight = newCtxt.lookup(self.operatorRepresentation['weight'])
        weight.values = np.transpose(weight.values, list(range(len(weight.shape)))[1:] + [0])
        return newCtxt, True


class CMSISConv2DParser(RQSConv2DParser):

    def __init__(self):
        super().__init__()
        assert self.input_sym_names is not None, "Assuming input_sym_names are used."
        self.input_sym_names.append('shift')

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        return all([
            self.operatorRepresentation['group'] == 1,
            # Make sure padding is square
            self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][2],
            self.operatorRepresentation['pads'][1] == self.operatorRepresentation['pads'][3],
            self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][1],
        ])


class CMSISDWConv1DParser(RQSConv1DParser):

    def __init__(self):
        super().__init__()
        assert self.input_sym_names is not None, "Assuming we are using input_sym_names"
        self.input_sym_names.append('shift')

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        # Make sure padding is square
        return self.operatorRepresentation['pads'][0] == self.operatorRepresentation['pads'][1]

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, wellFormed = super().parseNodeCtxt(ctxt, node, channels_first)
        if not wellFormed:
            return ctxt, False

        if not self.operatorRepresentation['group'] == self.operatorRepresentation['weight_shape'][-1]:
            return ctxt, False

        return newCtxt, True


class CMSISConv1DParser(RQSConv1DParser):

    def __init__(self):
        super().__init__()
        assert self.input_sym_names is not None, "Assuming we are using input_sym_names"
        self.input_sym_names.append('shift')


class CMSISLinearParser(GEMMParser):

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, wellFormed = super().parseNodeCtxt(ctxt, node, channels_first)
        if not wellFormed:
            return ctxt, False

        # Try to transpose A offline if possible, else fail
        if self.operatorRepresentation['transA'] == 1:
            nameA = self.operatorRepresentation['A']
            if newCtxt.is_global(nameA) and isinstance(newCtxt.lookup(nameA), ConstantBuffer):
                A = newCtxt.lookup(nameA)
                npA = np.asarray(A.values).reshape(A.shape)
                newA = np.transpose(npA, list(range(len(A.shape) - 2)) + [len(A.shape) - 1, len(A.shape) - 2])
                newCtxt.globalObjects[nameA].shape = newA.shape
                newCtxt.globalObjects[nameA].values = newA
                self.operatorRepresentation['transA'] = 0
            else:
                return newCtxt, False

        # Try to transpose B offline if possible, else fail
        # SCHEREMO: Magic trick - CMSIS works a bit weirdly with matmuls...
        if self.operatorRepresentation['transB'] == 0:
            nameB = self.operatorRepresentation['B']
            if newCtxt.is_global(nameB) and isinstance(newCtxt.lookup(nameB), ConstantBuffer):
                B = newCtxt.lookup(nameB)
                npB = np.asarray(B.values).reshape(B.shape)
                newB = np.transpose(npB, list(range(len(B.shape) - 2)) + [len(B.shape) - 1, len(B.shape) - 2])
                newCtxt.globalObjects[nameB].values = newB
                newCtxt.globalObjects[nameB].shape = newB.shape
                self.operatorRepresentation['transB'] = 1
            else:
                return newCtxt, False

        # Try to scale A offline if possible, else fail
        if self.operatorRepresentation['alpha'] != 1.0:
            nameA = self.operatorRepresentation['A']
            if newCtxt.is_global(nameA) and isinstance(newCtxt.lookup(nameA), ConstantBuffer):
                A = newCtxt.lookup(nameA)
                npA = np.asarray(A.values).reshape(A.shape)
                newA = npA * self.operatorRepresentation['beta']
                newCtxt.globalObjects[nameA].values = newA
                self.operatorRepresentation['alpha'] = 1.0
            else:
                return newCtxt, False

        # Try to scale B offline if possible, else fail
        if self.operatorRepresentation['beta'] != 1.0:
            nameB = self.operatorRepresentation['B']
            if newCtxt.is_global(nameB) and isinstance(newCtxt.lookup(nameB), ConstantBuffer):
                B = newCtxt.lookup(nameB)
                npB = np.asarray(B.values).reshape(B.shape)
                newB = npB * self.operatorRepresentation['beta']
                newCtxt.globalObjects[nameB].values = newB
                self.operatorRepresentation['beta'] = 1.0
            else:
                return newCtxt, False

        return newCtxt, True


class CMSISGEMMParser(CMSISLinearParser, RQSParserInterface):

    def __init__(self):
        CMSISLinearParser.__init__(self, noBiasHoisting = True)
        RQSParserInterface.__init__(self, "mul add")
        if self.required_attrs is not None:
            self.required_attrs.append('shift')
        else:
            self.required_attrs = ['shift']

    def parseNode(self, node: gs.Node) -> bool:
        if all([
                CMSISLinearParser.parseNode(self, node),
                RQSParserInterface.parseNode(self, node),
        ]):
            return False

        self.operatorRepresentation['shift'] = int(self._unpack_const(self.operatorRepresentation['shift']))
        return True


class CMSISMHSAParser(MHSAParser):

    def __init__(self):
        super().__init__()
        required_attrs = ['isoftmaxA', 'isoftmaxB', 'isoftmaxC', 'isoftmaxlog2']
        if self.required_attrs is not None:
            self.required_attrs.extend(required_attrs)
        else:
            self.required_attrs = required_attrs
        optional_attrs = {'signed': 1}
        if self.optional_attrs is not None:
            self.optional_attrs.update(optional_attrs)
        else:
            self.optional_attrs = optional_attrs

    def parseNode(self, node: gs.Node) -> bool:
        if not super().parseNode(node):
            return False

        self.operatorRepresentation.update({
            attr: self._int_unpack_attr(attr) for attr in [
                'preattn_requant_shift',
                'postattn_requant_shift',
                'wo_requant_shift',
                'wq_requant_shift',
                'wk_requant_shift',
                'wv_requant_shift',
                'isoftmaxA',
                'isoftmaxB',
                'isoftmaxC',
            ]
        })

        self.operatorRepresentation.update({
            attr: self._log2_int_unpack_attr(attr) for attr in [
                'preattn_requant_div',
                'postattn_requant_div',
                'wo_requant_div',
                'wq_requant_div',
                'wk_requant_div',
                'wv_requant_div',
                'isoftmaxlog2',
            ]
        })

        return True


class CMSISLinearAttentionParser(LinearAttentionParser):

    def __init__(self):
        super().__init__()
        optional_attrs = {'signed': 1}
        if self.optional_attrs is not None:
            self.optional_attrs.update(optional_attrs)
        else:
            self.optional_attrs = optional_attrs


class CMSISCLCAParser(CLCAParser):

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, wellFormed = super().parseNodeCtxt(ctxt, node, channels_first)
        if not wellFormed:
            return ctxt, False

        # Div to shift:
        self.operatorRepresentation.update({
            attr: self._log2_int_unpack_attr(attr) for attr in [
                'wq_requant_div',
                'wk_requant_div',
                'wv_requant_div',
                'wo_requant_div',
                'kdiv_requant_div',
                'preattn_requant_div',
                'postattn_requant_div',
            ]
        })

        # Fold additions:
        def add_fold(bias, add, mul):
            return bias + add // mul

        self.operatorRepresentation['wo_bias'] = add_fold(
            self._unpack_const(self.operatorRepresentation['wo_bias']),
            self._unpack_const(self.operatorRepresentation['wo_requant_add']),
            self._unpack_const(self.operatorRepresentation['wo_requant_mul']))

        self.operatorRepresentation['wq_bias'] = add_fold(
            self._unpack_const(self.operatorRepresentation['wq_bias']),
            self._unpack_const(self.operatorRepresentation['wq_requant_add']),
            self._unpack_const(self.operatorRepresentation['wq_requant_mul']))

        self.operatorRepresentation['wk_bias'] = add_fold(
            self._unpack_const(self.operatorRepresentation['wk_bias']),
            self._unpack_const(self.operatorRepresentation['wv_requant_add']),
            self._unpack_const(self.operatorRepresentation['wv_requant_mul']))

        # Rescale requant adds:
        def add_rescale(add, mul):
            return add // mul

        self.operatorRepresentation.update({
            f'{name}_requant_add':
                add_rescale(
                    self._unpack_const(self.operatorRepresentation[f'{name}_requant_add']),
                    self._unpack_const(self.operatorRepresentation[f'{name}_requant_mul']),
                ) for name in ['postattn', 'preattn', 'kdiv', 'wk', 'wo', 'wq', 'wv']
        })

        # Delta into mul
        kdiv_requant_mul = self._unpack_const(self.operatorRepresentation['kdiv_requant_mul'])
        Delta = self._unpack_const(self.operatorRepresentation['Delta'])
        self.operatorRepresentation['kdiv_requant_mul'] = kdiv_requant_mul * Delta

        return newCtxt, True
