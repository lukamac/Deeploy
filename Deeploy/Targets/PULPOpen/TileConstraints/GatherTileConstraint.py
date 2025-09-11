# ----------------------------------------------------------------------
#
# File: ConcatTileConstraint.py
#
# Last edited: 19.02.2024
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

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class GatherTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBuffer = ctxt.lookup(parseDict["data_in"])
        indicesBuffer = ctxt.lookup(parseDict["indices"])
        outputBuffer = ctxt.lookup(parseDict["data_out"])

        for buff in [inputBuffer, outputBuffer]:
            tilerModel.addTensorDimToModel(ctxt, buff.name)

        assert isinstance(indicesBuffer, ConstantBuffer)
        index = indicesBuffer.values.item()
        axis = parseDict["axis"]

        for i in range(axis):
            inVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = i)
            outVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = i)
            tilerModel.addConstraint(inVar == outVar)

        for i in range(axis + 1, len(inputBuffer.shape)):
            inVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = i)
            outVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = i - 1)
            tilerModel.addConstraint(inVar == outVar)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBuffer = ctxt.lookup(parseDict["data_in"])
        for i, dim in enumerate(inputBuffer.shape):
            var = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = i)
            tilerModel.addConstraint(var == dim)
        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:

        symbolicParseDict = parseDict.copy()

        return symbolicParseDict

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [absCube.rectangle for absCube in absoluteOutputCubes]

        indicesBuffer = ctxt.lookup(operatorRepresentation["indices"])
        assert isinstance(indicesBuffer, ConstantBuffer)
        index = indicesBuffer.values.item()
        axis = operatorRepresentation["axis"]

        inputBuffer = ctxt.lookup(operatorRepresentation["data_in"])

        inputCubes = []
        for cube in outputCubes:
            inputCubes.append(
                HyperRectangle(
                    offset = cube.offset[:axis] + (0,) + cube.offset[axis:],
                    dims = cube.dims[:axis] + (inputBuffer.shape[axis],) + cube.dims[axis:],
                ))

        inputLoadSchedule = [{"data_in": cube} for cube in inputCubes]
        outputLoadSchedule = [{"data_out": cube} for cube in outputCubes]

        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, ["data_in", "data_out"])

        schedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        repScheme = VariableReplacementScheme({}, {})

        return repScheme, schedule
