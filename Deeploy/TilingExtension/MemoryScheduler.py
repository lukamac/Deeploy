# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import _permute
from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, TransientBuffer, VariableBuffer
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy
from Deeploy.TilingExtension.MemoryConstraints import PatternMemoryConstraints, TensorMemoryConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel


@dataclass
class MemoryBlock:
    name: str
    level: str
    _lifetime: Tuple[int, int]
    _addrSpace: Optional[Tuple[int, int]] = None

    @property
    def addrSpace(self) -> Optional[Tuple[int, int]]:
        return self._addrSpace

    @addrSpace.setter
    def addrSpace(self, addrSpace: Optional[Tuple[int, int]]):
        if addrSpace is None:
            self._addrSpace = None
            return

        begin, end = addrSpace
        assert end >= begin, f"The end of the addres space should be greater or equal to the beginning. Received address space ({begin}, {end})"
        self._addrSpace = addrSpace

    @property
    def lifetime(self) -> Tuple[int, int]:
        return self._lifetime

    @lifetime.setter
    def lifetime(self, lifetime: Tuple[int, int]):
        begin, end = lifetime
        assert end >= begin, f"The end of lifetime should be greater or equal to the beginning. Received lifetime ({begin}, {end})"
        self._lifetime = lifetime

    def __init__(self, name: str, level: str, lifetime: Tuple[int, int], addrSpace: Optional[Tuple[int, int]]):
        self.name = name
        self.level = level
        self.lifetime = lifetime

        if addrSpace is not None:
            self.addrSpace = addrSpace

    def collides(self, other: MemoryBlock) -> bool:
        assert (isinstance(other, MemoryBlock)), f"{other} is not a MemoryBlock!"

        if self.addrSpace is None or other.addrSpace is None:
            return False

        xCollision: bool = False
        yCollision: bool = False

        if self.lifetime[0] <= other.lifetime[1] and self.lifetime[1] >= other.lifetime[0]:
            xCollision = True

        if self.addrSpace[0] < other.addrSpace[1] and self.addrSpace[1] > other.addrSpace[0]:
            yCollision = True

        return (xCollision and yCollision)


@dataclass
class Lifetime:
    begin: int
    end: int

    def contains(self, timestamp: int) -> bool:
        return self.begin <= timestamp and self.end >= timestamp

    def overlaps(self, other: Lifetime) -> bool:
        return self.contains(other.begin) or other.contains(self.begin)

    def offset(self, offset: int) -> Lifetime:
        return Lifetime(self.begin + offset, self.end + offset)


class MemoryScheduler():
    _ROWSUMNAME = "rowSum"
    _COLSUMNAME = "colSum"
    _PERMUTATIONIDXNAME = "permutationIdx"
    _INTERMEDIATEADJPRODUCTNAME = "intermediateAdjProduct"
    _FINALADJPRODUCTNAME = "AdjProduct"
    _COSTVARIABLENAME = "H"
    _COSTPRODUCTNAME = "costProduct"

    byteAlignment = 4

    @staticmethod
    def overlap(lifetimeA: Tuple[int, int], lifetimeB: Tuple[int, int]) -> bool:
        overlap: bool = False
        overlap |= (lifetimeA[0] >= lifetimeB[0] and lifetimeA[0] <= lifetimeB[1])
        overlap |= (lifetimeB[0] >= lifetimeA[0] and lifetimeB[0] <= lifetimeA[1])
        return overlap

    def __init__(self, stringSuffix: str, tileScheduler: bool, seed: int = 1996080121):
        self._stringSuffix = stringSuffix
        self.stringSuffix = ""
        self.tileScheduler = tileScheduler  # TODO: What is this?

        self.seed = seed
        self.memoryMap: Dict[str, List[List[MemoryBlock]]] = {}

        self._permutationState: Dict[str, Union[List[List[Union[IntVar]]], np.ndarray]] = {}

    def _addPermutationMatrix(self, tilerModel: TilerModel, numVars: int,
                              patternIdx: int) -> List[List[Union[IntVar, int]]]:

        permMat: List[List[Union[IntVar, int]]] = []

        for i in range(numVars):
            rowSumName = f"{self._ROWSUMNAME}_{i}" + self.stringSuffix
            jSum = tilerModel.addVariable(rowSumName, 0, 1, patternIdx)
            permMat.append([])
            for j in range(numVars):
                name = f"{self._PERMUTATIONIDXNAME}_{i}_{j}" + self.stringSuffix
                jVar = tilerModel.addVariable(name, 0, 1, patternIdx)
                permMat[i].append(jVar)
            tilerModel.addConstraint(tilerModel._model.SumEquality(permMat[i], jSum))
            tilerModel.addConstraint(jSum == 1)

        for i in range(numVars):
            colSumName = f"{self._COLSUMNAME}_{i}" + self.stringSuffix
            jSum = tilerModel.addVariable(colSumName, 0, 1, patternIdx)
            constraintVec = []
            for j in range(numVars):
                name = f"{self._PERMUTATIONIDXNAME}_{j}_{i}" + self.stringSuffix
                jVar = tilerModel.getVariable(name, patternIdx)
                constraintVec.append(jVar)
            tilerModel.addConstraint(tilerModel._model.SumEquality(constraintVec, jSum))
            tilerModel.addConstraint(jSum == 1)

        return permMat

    def _permuteMatrices(self, tilerModel: TilerModel, permutationMatrix: List[List[Union[IntVar, int]]],
                         adjacencyMatrix: List[List[int]], costVector: List[Union[int, IntVar]], patternIdx: int):

        def boolMatMulSingle(A, B, row, col, transposeB = False):

            constr = 0
            numVars = len(A)

            for j in range(numVars):
                if not transposeB:
                    constr += A[row][j] * B[j][col]
                else:
                    constr += A[row][j] * B[col][j]

            return constr

        def boolMatVecMulSingle(A, B, row):

            constr = 0
            numVars = len(B)

            for j in range(numVars):
                constr += A[row][j] * B[j]

            return constr

        permAdj_intermediate: List[List[Union[IntVar, int]]] = []
        permAdj: List[List[Union[IntVar, int]]] = []
        permCost: List[Union[IntVar, int]] = []

        numVars = len(costVector)

        for i in range(numVars):
            permAdj_intermediate.append([])
            for j in range(numVars):
                name = f"{self._INTERMEDIATEADJPRODUCTNAME}_{i}_{j}" + self.stringSuffix
                jVar = tilerModel.addVariable(name, 0, 1, patternIdx)
                constr = boolMatMulSingle(permutationMatrix, adjacencyMatrix, i, j, False)
                tilerModel.addConstraint(jVar == constr)
                permAdj_intermediate[i].append(jVar)

        for i in range(numVars):
            permAdj.append([])
            for j in range(numVars):
                name = f"{self._FINALADJPRODUCTNAME}_{i}_{j}" + self.stringSuffix
                jVar = tilerModel.addVariable(name, 0, 1, patternIdx)
                constr = boolMatMulSingle(permAdj_intermediate, permutationMatrix, i, j, True)
                tilerModel.addConstraint(jVar == constr)
                permAdj[i].append(jVar)

        costMax = 0
        for cost in costVector:
            if isinstance(cost, int):
                newCost = cost
            else:
                newCost = cost.Max()
            costMax = max(costMax, newCost)

        for j in range(numVars):
            name = f"{self._COSTPRODUCTNAME}_{j}" + self.stringSuffix
            jVar = tilerModel.addVariable(name, 0, costMax, patternIdx)
            constr = boolMatVecMulSingle(permutationMatrix, costVector, j)
            tilerModel.addConstraint(jVar == constr)
            permCost.append(jVar)

        return permAdj, permCost

    def _generateCost(self, tilerModel: TilerModel, adjMatrix: List[List[Union[int, IntVar]]],
                      costVector: List[Union[int, IntVar]], patternIdx: int):

        def maxVal(val) -> int:
            if isinstance(val, int):
                return val
            else:
                return val.Max()

        hVector = []
        numVars = len(costVector)

        name = f"{self._COSTVARIABLENAME}_0" + self.stringSuffix

        hVar = tilerModel.addVariable(name, 0, maxVal(costVector[0]), patternIdx)
        constr = hVar == costVector[0]
        tilerModel.addConstraint(constr)
        hVector.append(hVar)

        for i in range(1, numVars):
            name = f"{self._COSTVARIABLENAME}_{i}" + self.stringSuffix
            # SCHEREMO: Check for overlap here!
            hVar = tilerModel.addVariable(name, 0, maxVal(hVector[i - 1]) + maxVal(costVector[i]), patternIdx)
            prod = []
            for j in range(i):
                name = f"{self._COSTVARIABLENAME}_{i}_maxEntry_{j}" + self.stringSuffix
                pVar = tilerModel.addVariable(name, 0, maxVal(hVector[j]) + maxVal(costVector[i]), patternIdx)
                constr = (pVar == (adjMatrix[i][j] * hVector[j] + costVector[i]))
                tilerModel.addConstraint(constr)
                prod.append(pVar)
            tilerModel.addConstraint(tilerModel._model.MaxEquality(prod, hVar))
            hVector.append(hVar)

        name = "cost" + self.stringSuffix
        costMax = max([maxVal(entry) for entry in hVector])
        cost = tilerModel.addVariable(name, 0, costMax, patternIdx)
        tilerModel.addConstraint(tilerModel._model.MaxEquality(hVector, cost))

        return cost

    def _buildInterferenceGraph(self, lifetimeMap) -> Dict[str, List[str]]:

        interferenceGraph: Dict[str, List[str]] = {}
        for name, lifetime in lifetimeMap.items():
            neighbors: List[str] = []
            for neighborName, neighborLifetime in lifetimeMap.items():
                if neighborName == name:
                    continue

                if self.overlap(lifetime, neighborLifetime):
                    neighbors.append(neighborName)

            interferenceGraph[name] = neighbors

        return interferenceGraph

    def _calculateLifetimes(self, ctxt: NetworkContext, patternMemoryConstraint: PatternMemoryConstraints,
                            memoryLevel: str) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, TensorMemoryConstraint]]:

        def filterBuffers(buffer: VariableBuffer) -> bool:
            if not buffer._deploy:
                return False

            # SCHEREMO: Transient buffers are only considered by last-level schedulers
            if isinstance(buffer, TransientBuffer):
                return self.tileScheduler
            elif isinstance(buffer, ConstantBuffer):
                return self.tileScheduler and (memoryLevel != buffer._memoryLevel)
            else:
                return self.tileScheduler ^ (memoryLevel == buffer._memoryLevel)

        tensorMap = OrderedDict()
        lifetimeMap: Dict[str, Tuple[int, int]] = dict()
        maxStepIdx = len(patternMemoryConstraint.nodeConstraints)

        for stepIdx, nodeConstraint in enumerate(patternMemoryConstraint.nodeConstraints):
            for tensorName, tensorMemoryConstraint in nodeConstraint.tensorMemoryConstraints.items():
                if memoryLevel not in tensorMemoryConstraint.memoryConstraints:
                    continue

                buffer = ctxt.lookup(tensorName)
                assert isinstance(buffer, VariableBuffer)

                if not filterBuffers(buffer):
                    continue

                # LMACAN: Update end of lifetime for all existing aliases to the current one
                #         because that one is the oldest one so far
                for alias in ctxt.allAliases(tensorName):
                    if alias in lifetimeMap:
                        start = lifetimeMap[alias][0]
                        lifetimeMap[alias] = (start, stepIdx)

                if tensorName in lifetimeMap:
                    start = lifetimeMap[tensorName][0]
                    lifetimeMap[tensorName] = (start, stepIdx)
                else:
                    lifetimeMap[tensorName] = (stepIdx, stepIdx)
                    tensorMap[tensorName] = tensorMemoryConstraint

        # JUNGVI: Align the lifetime of I/O tensors accordignly:
        #   - Input Tensors are alive at step 0
        #   - Output Tensors are alive until the last step
        # TODO: Why do we have to fixup the lifetime, i.e. why doesn't the top loop do it?
        for tensorName, lifetime in lifetimeMap.items():
            buffer = ctxt.lookup(tensorName)
            assert isinstance(buffer, VariableBuffer)
            if buffer.is_input:
                lifetimeMap[tensorName] = (0, lifetime[-1])
            elif buffer.is_output:
                lifetimeMap[tensorName] = (lifetime[0], maxStepIdx)

        return lifetimeMap, tensorMap

    def _buildAdjacencyMatrix(self, graph, tensorMap):
        numVars = len(graph)

        adjacencyMatrix = np.zeros((numVars, numVars), dtype = int)

        for node, neighbors in graph.items():
            nodeIdx = list(tensorMap.keys()).index(node)
            for neighbor in neighbors:
                adjacencyIdx = list(tensorMap.keys()).index(neighbor)
                adjacencyMatrix[nodeIdx, adjacencyIdx] = 1

        return adjacencyMatrix

    def _buildCostVector(self, ctxt: NetworkContext, graph, tensorMap: Dict[str, TensorMemoryConstraint], memoryLevel):
        costVector: List[Union[int, IntVar]] = []

        for tensor, neighbors in graph.items():
            constr = tensorMap[tensor].memoryConstraints[memoryLevel]

            buffer = ctxt.lookup(tensor)
            assert isinstance(buffer, VariableBuffer)

            # LMACAN: Alias buffers are costless when the buffer they alias is a neighbor
            if buffer.aliasedBuffer is not None and buffer.aliasedBuffer in neighbors:
                costVector.append(0)
                continue

            # TODO: This should be a method in either buffer, _type (Pointer), or referencedType (ImmediateType)
            # NOTE: For now assume types are divisible by 8
            assert buffer._type.referencedType.typeWidth % 8 == 0
            sizeInBytes = constr.size * (buffer._type.referencedType.typeWidth // 8)

            # SCHEREMO: Make sure each tile is word-aligned for better access performance
            # and to comply with implicit PULP L3 tiling bugs
            sizeInBytesAligned = ((sizeInBytes + self.byteAlignment - 1) // self.byteAlignment) * self.byteAlignment
            costVector.append(sizeInBytesAligned * constr.multiBufferCoefficient)

        if len(costVector) == 0:
            costVector.append(0)

        return costVector

    def heuristicPermutation(self, adjacencyMatrix, costVector) -> List[int]:
        permutationList = list(range(len(costVector)))
        random.seed(self.seed)
        random.shuffle(permutationList)

        return permutationList

    def _stablePermutation(self, adjacencyMatrix, costVector, permutationList):

        if len(costVector) == 1:
            return adjacencyMatrix, costVector, np.ones_like(adjacencyMatrix)

        permutationMatrix = np.zeros_like(adjacencyMatrix)
        newCostVector = []

        for i in permutationList:
            newCostVector.append(costVector[i])

        for idx, i in enumerate(permutationList):
            permutationMatrix[idx, i] = 1

        newAdjacencyMatrix = permutationMatrix @ adjacencyMatrix @ np.transpose(permutationMatrix)

        return newAdjacencyMatrix, newCostVector, permutationMatrix

    # SCHEREMO: Set the end of the lifetime of in-place operator inputs to the lifetime of their outputs
    def _dealiasLifetimeMap(self, ctxt: NetworkContext,
                            tensorLifetimeMap: Dict[str, Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:

        tensorLifetimeMap = tensorLifetimeMap.copy()

        if not self.tileScheduler:
            for key, lifetime in tensorLifetimeMap.items():
                alias = ctxt.dealiasBuffer(key)

                if alias == key:
                    continue

                if ctxt.is_global(alias):
                    tensorLifetimeMap[key] = (0, lifetime[1])
                    continue

                aliasLifetime = tensorLifetimeMap[alias]
                tensorLifetime = (aliasLifetime[0], max(aliasLifetime[1], lifetime[1]))
                tensorLifetimeMap[alias] = tensorLifetime

        return tensorLifetimeMap

    def getConstantTensorOffset(self, ctxt: NetworkContext, memoryLevel: str):
        constantTensorSize = 0
        for buffer in ctxt.globalObjects.values():
            if not "MEMORYARENA" in buffer.name and isinstance(buffer,
                                                               ConstantBuffer) and buffer._memoryLevel == memoryLevel:
                constantTensorSize += np.prod(buffer.shape) * buffer._type.referencedType.typeWidth // 8

        return int(constantTensorSize)

    def _scheduleMemoryConstraints(self,
                                   tilerModel: TilerModel,
                                   ctxt: NetworkContext,
                                   patternMemoryConstraints: List[PatternMemoryConstraints],
                                   memoryHierarchy: MemoryHierarchy,
                                   memoryAllocStrategy: Literal["TetrisRandom", "TetrisCo-Opt"],
                                   memoryLevel: str = "L1"):

        if memoryLevel not in self.memoryMap:
            self.memoryMap[memoryLevel] = []

        for patternIdx, patternMemoryConstraint in enumerate(patternMemoryConstraints):
            tensorLifetimeMap, tensorMap = self._calculateLifetimes(ctxt, patternMemoryConstraint, memoryLevel)

            #missingTensors = [
            #    tensorMc.tensorName for nodeConstr in patternMemoryConstraint.nodeConstraints
            #    for tensorMc in nodeConstr.tensorMemoryConstraints.values()
            #    if ctxt.lookup(tensorMc.tensorName)._deploy and tensorMc.tensorName not in tensorMap
            #]
            #assert len(missingTensors) == 0, f"Some tensors have not been assigned their memory constraint: {missingTensors}"

            tensorLifetimeMap = self._dealiasLifetimeMap(ctxt, tensorLifetimeMap)

            interferenceGraph = self._buildInterferenceGraph(tensorLifetimeMap)

            numVars = len(interferenceGraph)

            adjacencyMatrix = self._buildAdjacencyMatrix(interferenceGraph, tensorMap)
            costVector = self._buildCostVector(ctxt, interferenceGraph, tensorMap, memoryLevel)

            blocks = []
            for tensor in interferenceGraph.keys():
                relativeLifeTime = tensorLifetimeMap[tensor]
                lifetime = (relativeLifeTime[0] + patternIdx, relativeLifeTime[1] + patternIdx)
                blocks.append(MemoryBlock(tensor, memoryLevel, lifetime, None))

            self.memoryMap[memoryLevel].append(blocks)

            # SCHEREMO: Build permutation matrix
            if memoryAllocStrategy == 'TetrisCo-Opt':
                if numVars > 1:
                    permutationMatrix = self._addPermutationMatrix(tilerModel, numVars, patternIdx)
                    permAdj, permCost = self._permuteMatrices(tilerModel, permutationMatrix, adjacencyMatrix,
                                                              costVector, patternIdx)
                else:
                    permutationMatrix = np.ones((1,))
                    permAdj, permCost = adjacencyMatrix, costVector

            elif memoryAllocStrategy == 'TetrisRandom':
                permutationList = self.heuristicPermutation(adjacencyMatrix, costVector)
                permAdj, permCost, permutationMatrix = self._stablePermutation(adjacencyMatrix, costVector,
                                                                               permutationList)
            elif memoryAllocStrategy == "MiniMalloc":
                #JUNVI: When using MiniMalloc we don't perform memory allocation with Tiling, hence we don't add the permutation constraints
                continue
            else:
                raise (f"Unrecognized memory allocation strategy {memoryAllocStrategy}!")

            self._permutationState[memoryLevel + f"_{patternIdx}"] = permutationMatrix

            constantTensorOffset = self.getConstantTensorOffset(ctxt, memoryLevel)

            cost = self._generateCost(tilerModel, permAdj, permCost, patternIdx)
            constr = (cost + constantTensorOffset) < memoryHierarchy.memoryLevels[memoryLevel].size
            tilerModel.addConstraint(constr)

        return

    def scheduleMemoryConstraints(self,
                                  tilerModel: TilerModel,
                                  ctxt: NetworkContext,
                                  allMemoryConstraints: List[PatternMemoryConstraints],
                                  memoryHierarchy: MemoryHierarchy,
                                  memoryAllocStrategy: Literal["TetrisRandom", "TetrisCo-Opt"],
                                  memoryLevel: str = "L1"):

        self.stringSuffix = self._stringSuffix + f"_{memoryLevel}"
        return self._scheduleMemoryConstraints(tilerModel, ctxt, allMemoryConstraints, memoryHierarchy,
                                               memoryAllocStrategy, memoryLevel)

    def constraintTileBuffersWithOverlappingLifetime(self, tilerModel: TilerModel, ctxt: NetworkContext,
                                                     patternMemoryConstraint: PatternMemoryConstraints,
                                                     memoryHierarchy: MemoryHierarchy):
        """This method adds the necessary constraints for tiling to be performed before the static memory allocation of the tile buffers.
        To perform static memory allocation after tiling (i.e. decouple tiling and memory alloc), we need to do two assumptions

            1. All tile buffers for each node have overlapping lifetime, so we can find their memory footprint by just summing their sizes and hence we don't need to know the specific memory allocation. This assumption is true as soon as we don't do tile several nodes together (ask me if you don't know what I mean here).
            2. We don't allocate the tensors of the graph in the same memory level than the tiles (for instance we put all tensor in L2 and the tiles only live in L1).
        """

        for nodeConstraint in patternMemoryConstraint.nodeConstraints:
            tileMemoryConstraint = {}

            for tensorMemoryConstraints in nodeConstraint.tensorMemoryConstraints.values():
                for memoryConstraint in tensorMemoryConstraints.memoryConstraints.values():
                    if isinstance(memoryConstraint.size, IntVar):

                        _buffer = ctxt.lookup(tensorMemoryConstraints.tensorName)

                        if not isinstance(_buffer, TransientBuffer):
                            _typeWidthFactor = int(_buffer._type.referencedType.typeWidth / 8)
                        else:
                            _typeWidthFactor = 1

                        tileMemoryConstraint[tensorMemoryConstraints.tensorName] = {
                            "sizeVar": memoryConstraint.size,
                            "typeWidthFactor": _typeWidthFactor,
                            "memoryLevel": memoryConstraint.memoryLevel,
                            "multiBufferCoeff": memoryConstraint.multiBufferCoefficient,
                        }

            for memoryLevel in memoryHierarchy.memoryLevels.values():
                sumExpr = 0
                constantTensorOffset = self.getConstantTensorOffset(ctxt, memoryLevel.name)
                for infoDict in tileMemoryConstraint.values():
                    if memoryLevel.name == infoDict['memoryLevel']:
                        sumExpr += infoDict['sizeVar'] * infoDict['typeWidthFactor'] * infoDict['multiBufferCoeff']
                if sumExpr != 0:
                    tilerModel.addConstraint(sumExpr + constantTensorOffset, memoryLevel = memoryLevel)

    def getSymbolicCostName(self, patternIdx: int, memoryLevel: str) -> str:
        stringSuffix = self._stringSuffix + f"_{memoryLevel}"

        name = f"cost{stringSuffix}"
        return name

    def getCost(self, tilerModel, patternIdx: int, memoryLevel: str) -> int:

        stringSuffix = self._stringSuffix + f"_{memoryLevel}"

        name = f"cost{stringSuffix}_copyIdx_{patternIdx}"
        symVar = tilerModel._variables[name]
        var = tilerModel._resolveVariable(symVar)
        cost = var

        return cost

    def getHVector(self, tilerModel, patternIdx: int, memoryLevel: str) -> np.ndarray:

        stringSuffix = self._stringSuffix + f"_{memoryLevel}"
        numVars = len(self.memoryMap[memoryLevel][patternIdx])

        hVec = np.zeros((numVars))

        for i in range(numVars):
            name = f"{self._COSTVARIABLENAME}_{i}{stringSuffix}_copyIdx_{patternIdx}"
            symVar = tilerModel._variables[name]
            var = tilerModel._resolveVariable(symVar)
            hVec[i] = var

        return hVec

    def getBlockVector(self, patternIdx: int, memoryLevel: str) -> List[MemoryBlock]:

        return self.memoryMap[memoryLevel][patternIdx]

    def getPMatrix(self, tilerModel, patternIdx: int, memoryLevel: str) -> np.ndarray:

        stringSuffix = self._stringSuffix + f"_{memoryLevel}"
        numVars = len(self.memoryMap[memoryLevel][patternIdx])
        permMat = np.zeros((numVars, numVars))

        for i in range(numVars):
            for j in range(numVars):
                name = f"{self._PERMUTATIONIDXNAME}_{i}_{j}{stringSuffix}_copyIdx_{patternIdx}"
                symVar = tilerModel._variables[name]
                var = tilerModel._resolveVariable(symVar)
                permMat[i, j] = var

        return permMat

    def annotateSolution(self, ctxt: NetworkContext, tilerModel: TilerModel):

        def permMatrix2permList(permMatrix: np.ndarray) -> List[int]:

            _permMatrix = []

            if len(permMatrix) == 0:
                return []

            if len(permMatrix) == 1:
                return [0]

            for i in range(permMatrix.shape[0]):
                rowVec = list(permMatrix[i])
                _permMatrix.append(rowVec)

            return [row.index(1) for row in _permMatrix]

        for memoryLevel, patternList in self.memoryMap.items():
            for patternIdx, pattern in enumerate(patternList):

                permutationMatrix = self._permutationState[memoryLevel + f"_{patternIdx}"]

                if not isinstance(permutationMatrix, np.ndarray):
                    _permutationMatrix = self.getPMatrix(tilerModel, patternIdx, memoryLevel)
                else:
                    _permutationMatrix = permutationMatrix

                permList = permMatrix2permList(_permutationMatrix)

                if pattern != [] and len(pattern) > 1:
                    permPattern = _permute(pattern, permList)
                else:
                    permPattern = pattern

                aliasedBlocks = []

                for blockIdx, memoryBlock in enumerate(permPattern):

                    blockNames = [block.name for block in permPattern]
                    buffer = ctxt.lookup(memoryBlock.name)
                    assert isinstance(buffer, VariableBuffer)

                    isAliasToGlobal = any(ctxt.is_global(alias) for alias in buffer.aliases)

                    # SCHEREMO: If we're handling an active alias to a global buffer in their home memory level, we don't need to resolve addresses
                    if isAliasToGlobal and buffer._memoryLevel == memoryLevel:
                        continue

                    # SCHEREMO: Don't fully unroll aliases here - this is pattern-sensitive!
                    buffAliasesInBlockNames = [alias for alias in buffer.aliases if alias in blockNames]
                    aliasedBlocks.extend([(memoryBlock, alias) for alias in buffAliasesInBlockNames])
                    if len(buffAliasesInBlockNames) > 0:
                        continue

                    upperIdx = blockIdx

                    upperEndVar = tilerModel.getVariable(
                        f"{self._COSTVARIABLENAME}_{upperIdx}{self._stringSuffix}_{memoryLevel}", patternIdx)
                    upperEnd = tilerModel._resolveVariable(upperEndVar)

                    maxAddr = 0
                    for idx, oldBlock in enumerate(permPattern):
                        if self.overlap(oldBlock.lifetime, memoryBlock.lifetime):
                            if oldBlock.addrSpace is not None:
                                maxAddr = max(maxAddr, oldBlock.addrSpace[1])

                    lowerEnd = maxAddr
                    memoryBlock.addrSpace = (lowerEnd, upperEnd)

                for block, alias in aliasedBlocks:
                    for refBlock in sorted(permPattern, key = lambda x: x.lifetime[0]):
                        if refBlock.name == alias:
                            block.addrSpace = refBlock.addrSpace
                            break
