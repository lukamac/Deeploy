# ----------------------------------------------------------------------
#
# File: Tiler.py
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


from Deeploy.Targets.Neureka.TileConstraints.NeurekaDenseConstraint import NeurekaRQSDenseConv2DTileConstraint, \
    NeurekaWmemRQSDenseConv2DTileConstraint
from Deeploy.Targets.Neureka.TileConstraints.NeurekaDepthwiseConstraint import NeurekaRQSDWConv2DTileConstraint, \
    NeurekaWmemRQSDWConv2DTileConstraint
from Deeploy.Targets.Neureka.TileConstraints.NeurekaPointwiseConstraint import NeurekaRQSPWConv2DTileConstraint, \
    NeurekaWmemRQSPWConv2DTileConstraint
from Deeploy.Targets.NeurekaV2.Bindings import NeurekaV2RQSDenseConv2DBindings, NeurekaV2RQSDWConv2DBindings, \
    NeurekaV2RQSPWConv2DBindings, NeurekaV2WmemRQSDenseConv2DBindings, NeurekaV2WmemRQSDWConv2DBindings, \
    NeurekaV2WmemRQSPWConv2DBindings
from Deeploy.TilingExtension.TilerExtension import TilingReadyNodeBindings

NeurekaV2RQSPWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NeurekaV2RQSPWConv2DBindings,
                                                                  tileConstraint = NeurekaRQSPWConv2DTileConstraint())
#NeurekaV2PWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NeurekaV2PWConv2DBindings,
#                                                               tileConstraint = NeurekaPWConv2DTileConstraint())

NeurekaV2WmemRQSPWConv2DTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = NeurekaV2WmemRQSPWConv2DBindings, tileConstraint = NeurekaWmemRQSPWConv2DTileConstraint())
#NeurekaV2WmemPWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NeurekaV2WmemPWConv2DBindings,
#                                                                   tileConstraint = NeurekaWmemPWConv2DTileConstraint())

NeurekaV2RQSDWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NeurekaV2RQSDWConv2DBindings,
                                                                  tileConstraint = NeurekaRQSDWConv2DTileConstraint())
#NeurekaV2DWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NeurekaV2DWConv2DBindings,
#                                                               tileConstraint = NeurekaDWConv2DTileConstraint())

NeurekaV2WmemRQSDWConv2DTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = NeurekaV2WmemRQSDWConv2DBindings, tileConstraint = NeurekaWmemRQSDWConv2DTileConstraint())
#NeurekaV2WmemDWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NeurekaV2WmemDWConv2DBindings,
#                                                                   tileConstraint = NeurekaWmemDWConv2DTileConstraint())

NeurekaV2RQSDenseConv2DTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = NeurekaV2RQSDenseConv2DBindings, tileConstraint = NeurekaRQSDenseConv2DTileConstraint())
#NeurekaV2DenseConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NeurekaV2DenseConv2DBindings,
#                                                                  tileConstraint = NeurekaDenseConv2DTileConstraint())

NeurekaV2WmemRQSDenseConv2DTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = NeurekaV2WmemRQSDenseConv2DBindings, tileConstraint = NeurekaWmemRQSDenseConv2DTileConstraint())
#NeurekaV2WmemDenseConv2DTilingReadyBindings = TilingReadyNodeBindings(
#    nodeBindings = NeurekaV2WmemDenseConv2DBindings, tileConstraint = NeurekaWmemDenseConv2DTileConstraint())
