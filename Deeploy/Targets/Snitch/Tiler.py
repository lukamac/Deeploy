# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.Targets.Generic.TileConstraints.AddTileConstraint import AddTileConstraint
from Deeploy.Targets.Generic.Bindings import BasicFusedAddReluBinding, BasicFloatFusedConv2dReluBinding
from Deeploy.Targets.Snitch.Bindings import SnitchAddBindings, SnitchGemmBindings, SnitchiNoNormBindings, \
    SnitchiSoftmaxBindings, SnitchRQAddBindings, SnitchRqGemmBindings
from Deeploy.Targets.Snitch.TileConstraints import iNoNormTileConstraint, iSoftmaxTileConstraint
from Deeploy.Targets.Snitch.TileConstraints.GemmTileConstraint import GemmTileConstraint
from Deeploy.Targets.Snitch.TileConstraints.RqGemmTileConstraint import RqGemmTileConstraint
from Deeploy.Targets.Snitch.TileConstraints.ConvTileConstraint import ConvTileConstraint
from Deeploy.TilingExtension.TilerExtension import TilingReadyNodeBindings

SnitchiSoftmaxTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = SnitchiSoftmaxBindings,
                                                            tileConstraint = iSoftmaxTileConstraint())
SnitchiNoNormTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = SnitchiNoNormBindings,
                                                           tileConstraint = iNoNormTileConstraint())
SnitchRQAddTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = SnitchRQAddBindings,
                                                         tileConstraint = AddTileConstraint())
SnitchGemmTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = SnitchGemmBindings,
                                                        tileConstraint = GemmTileConstraint())
SnitchRqGemmTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = SnitchRqGemmBindings,
                                                          tileConstraint = RqGemmTileConstraint())
SnitchAddTileReadyBindings = TilingReadyNodeBindings(nodeBindings = SnitchAddBindings,
                                                     tileConstraint = AddTileConstraint())
SnitchFloatFusedAddReluTileReadyBindings = TilingReadyNodeBindings(nodeBindings=[BasicFusedAddReluBinding], tileConstraint=AddTileConstraint())
SnitchFloatFusedConv2dReluTileReadyBindings = TilingReadyNodeBindings(nodeBindings=[BasicFloatFusedConv2dReluBinding], tileConstraint=ConvTileConstraint())
