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


from Deeploy.Targets.NeurekaV2.Bindings import NeurekaV2DenseConv2DBindings, NeurekaV2DWConv2DBindings, \
    NeurekaV2PWConv2DBindings, NeurekaV2RQSDenseConv2DBindings, NeurekaV2RQSDWConv2DBindings, \
    NeurekaV2RQSPWConv2DBindings, NeurekaV2WmemDenseConv2DBindings, NeurekaV2WmemDWConv2DBindings, \
    NeurekaV2WmemPWConv2DBindings, NeurekaV2WmemRQSDenseConv2DBindings, NeurekaV2WmemRQSDWConv2DBindings, \
    NeurekaV2WmemRQSPWConv2DBindings
from Deeploy.Targets.NeurekaV2.TileConstraints.NeurekaV2DenseConstraint import NeurekaV2DenseConv2DTileConstraint, \
    NeurekaV2RQSDenseConv2DTileConstraint, NeurekaV2WmemDenseConv2DTileConstraint, \
    NeurekaV2WmemRQSDenseConv2DTileConstraint
from Deeploy.Targets.NeurekaV2.TileConstraints.NeurekaV2DepthwiseConstraint import NeurekaV2DWConv2DTileConstraint, \
    NeurekaV2RQSDWConv2DTileConstraint, NeurekaV2WmemDWConv2DTileConstraint, NeurekaV2WmemRQSDWConv2DTileConstraint
from Deeploy.Targets.NeurekaV2.TileConstraints.NeurekaV2PointwiseConstraint import NeurekaV2PWConv2DTileConstraint, \
    NeurekaV2RQSPWConv2DTileConstraint, NeurekaV2WmemPWConv2DTileConstraint, NeurekaV2WmemRQSPWConv2DTileConstraint
from Deeploy.TilingExtension.TilerExtension import TilingReadyNodeBindings

NeurekaV2RQSPWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NeurekaV2RQSPWConv2DBindings,
                                                                  tileConstraint = NeurekaV2RQSPWConv2DTileConstraint())
NeurekaV2PWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NeurekaV2PWConv2DBindings,
                                                               tileConstraint = NeurekaV2PWConv2DTileConstraint())

NeurekaV2WmemRQSPWConv2DTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = NeurekaV2WmemRQSPWConv2DBindings, tileConstraint = NeurekaV2WmemRQSPWConv2DTileConstraint())
NeurekaV2WmemPWConv2DTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = NeurekaV2WmemPWConv2DBindings, tileConstraint = NeurekaV2WmemPWConv2DTileConstraint())

NeurekaV2RQSDWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NeurekaV2RQSDWConv2DBindings,
                                                                  tileConstraint = NeurekaV2RQSDWConv2DTileConstraint())
NeurekaV2DWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NeurekaV2DWConv2DBindings,
                                                               tileConstraint = NeurekaV2DWConv2DTileConstraint())

NeurekaV2WmemRQSDWConv2DTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = NeurekaV2WmemRQSDWConv2DBindings, tileConstraint = NeurekaV2WmemRQSDWConv2DTileConstraint())
NeurekaV2WmemDWConv2DTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = NeurekaV2WmemDWConv2DBindings, tileConstraint = NeurekaV2WmemDWConv2DTileConstraint())

NeurekaV2RQSDenseConv2DTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = NeurekaV2RQSDenseConv2DBindings, tileConstraint = NeurekaV2RQSDenseConv2DTileConstraint())
NeurekaV2DenseConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NeurekaV2DenseConv2DBindings,
                                                                  tileConstraint = NeurekaV2DenseConv2DTileConstraint())

NeurekaV2WmemRQSDenseConv2DTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = NeurekaV2WmemRQSDenseConv2DBindings, tileConstraint = NeurekaV2WmemRQSDenseConv2DTileConstraint())
NeurekaV2WmemDenseConv2DTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = NeurekaV2WmemDenseConv2DBindings, tileConstraint = NeurekaV2WmemDenseConv2DTileConstraint())
