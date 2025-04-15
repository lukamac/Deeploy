# ----------------------------------------------------------------------
#
# File: TransposeTileConstraint.py
#
# Last edited: 01.06.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict

from Deeploy.DeeployTypes import NetworkContext
from Deeploy.Targets.Generic.TileConstraints.TransposeTileConstraint import \
    TransposeTileConstraint as GenericTransposeTileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel


class TransposeTileConstraint(GenericTransposeTileConstraint):

    # Override this
    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBuffer = ctxt.lookup(parseDict['data_in'])

        # Only 1 dim can be tiled at most because pulp currently supports at most 2d DMA transfers
        tilerModel.addLimitTiledBufferDimensionsConstraint(inputBuffer, 1)

        return tilerModel
