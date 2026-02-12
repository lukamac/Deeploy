# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random
from collections import OrderedDict
from typing import Dict, List, Literal, Sequence, Tuple, Union

import numpy as np
from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.CommonExtensions.PermutationUtils import _permute
from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, TransientBuffer, VariableBuffer
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy
from Deeploy.TilingExtension.MemoryConstraints import PatternMemoryConstraint, TensorMemoryConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingTypes import AddressSpace, Lifetime, MemoryBlock


class MemoryScheduler():
    _ROWSUMNAME = "rowSum"
    _COLSUMNAME = "colSum"
    _PERMUTATIONIDXNAME = "permutationIdx"
    _INTERMEDIATEADJPRODUCTNAME = "intermediateAdjProduct"
    _FINALADJPRODUCTNAME = "AdjProduct"
    _COSTVARIABLENAME = "H"
    _COSTPRODUCTNAME = "costProduct"

    byteAlignment = 4

    def __init__(self, stringSuffix: str, tileScheduler: bool, seed: int = 1996080121):
        self._stringSuffix = stringSuffix
        self.stringSuffix = ""
        self.tileScheduler = tileScheduler  # TODO: What is this?

        self.seed = seed
        self.memoryMap: Dict[str, List[List[MemoryBlock]]] = {}

        self._permutationState: Dict[str, Union[List[List[Union[IntVar]]], np.ndarray]] = {}

    def _transposeMatrix(self, x: List[List[IntVar]]) -> List[List[IntVar]]:
        return list(map(list, zip(*x, strict = True)))

    def _initVarMatrix(self, name: str, height: int, width: int, lowerBound: int, upperBound: int,
                       tilerModel: TilerModel, copyIdx: int) -> List[List[IntVar]]:
        return [[
            tilerModel.addVariable(f"{name}_{i}_{j}" + self.stringSuffix, lowerBound, upperBound, copyIdx)
            for j in range(width)
        ]
                for i in range(height)]

    def _initVarVector(self, name: str, length: int, lowerBound: int, upperBound: int, tilerModel: TilerModel,
                       copyIdx: int) -> List[IntVar]:
        return [
            tilerModel.addVariable(f"{name}_{i}" + self.stringSuffix, lowerBound, upperBound, copyIdx)
            for i in range(length)
        ]

    def _addPermutationMatrix(self, tilerModel: TilerModel, numVars: int, patternIdx: int) -> List[List[IntVar]]:
        # Create permutation matrix
        permMat = self._initVarMatrix(self._PERMUTATIONIDXNAME, numVars, numVars, 0, 1, tilerModel, patternIdx)

        # Constraint row sum to 1
        for i, row in enumerate(permMat):
            sumVar = tilerModel.addVariable(f"{self._ROWSUMNAME}_{i}" + self.stringSuffix, 0, 1, patternIdx)
            tilerModel.addConstraint(tilerModel._model.SumEquality(row), sumVar)
            tilerModel.addConstraint(sumVar == 1)

        # Constraint column sum to 1
        for i, col in enumerate(self._transposeMatrix(permMat)):
            sumVar = tilerModel.addVariable(f"{self._COLSUMNAME}_{i}" + self.stringSuffix, 0, 1, patternIdx)
            tilerModel.addConstraint(tilerModel._model.SumEquality(col), sumVar)
            tilerModel.addConstraint(sumVar == 1)

        return permMat

    def _addMatMulConstraint(self, A: Sequence[Sequence[IntVar]], B: Sequence[Sequence[IntVar]],
                             C: Sequence[Sequence[IntVar]], tilerModel: TilerModel):
        M = len(A)
        assert M > 0
        assert len(C) == M
        K = len(A[0])
        assert K > 0
        assert len(B) == K
        N = len(B[0])
        assert len(C[0]) == N

        for m in range(M):
            for n in range(N):
                sum = 0
                for k in range(K):
                    sum += A[m][k] * B[k][n]
                tilerModel.addConstraint(C[m][n] == sum)

    def _addMatVecMulConstraint(self, mat: Sequence[Sequence[IntVar]], vec: Sequence[IntVar], resVec: Sequence[IntVar],
                                tilerModel: TilerModel):
        M = len(mat)
        assert M > 0
        K = len(mat[0])
        assert K > 0
        assert len(vec) == K
        assert len(resVec) == M

        for m in range(M):
            sum = 0
            for k in range(K):
                sum += mat[m][k] * vec[k]
            tilerModel.addConstraint(resVec[m] == sum)

    def _permuteMatrices(self, tilerModel: TilerModel, permutationMatrix: List[List[Union[IntVar, int]]],
                         adjacencyMatrix: List[List[int]], costVector: List[Union[int, IntVar]], patternIdx: int):
        numVars = len(costVector)

        permAdj_intermediate = self._initVarMatrix(self._INTERMEDIATEADJPRODUCTNAME, numVars, numVars, 0, 1, tilerModel,
                                                   patternIdx)
        permAdj = self._initVarMatrix(self._FINALADJPRODUCTNAME, numVars, numVars, 0, 1, tilerModel, patternIdx)

        costMax = 0
        for cost in costVector:
            if isinstance(cost, int):
                newCost = cost
            else:
                newCost = cost.Max()
            costMax = max(costMax, newCost)
        permCost = self._initVarVector(self._COSTPRODUCTNAME, numVars, 0, costMax, tilerModel, patternIdx)

        self._addMatMulConstraint(permutationMatrix, adjacencyMatrix, permAdj_intermediate, tilerModel)
        self._addMatMulConstraint(permAdj_intermediate, list(map(list, zip(*permutationMatrix))), permAdj, tilerModel)
        self._addMatVecMulConstraint(permutationMatrix, costVector, permCost, tilerModel)

        return permAdj, permCost

    def _generateCost(self, tilerModel: TilerModel, adjMatrix: List[List[Union[int, IntVar]]],
                      costVector: List[Union[int, IntVar]], patternIdx: int):

        def maxVal(val: Union[int, IntVar]) -> int:
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

    def _buildInterferenceGraph(self, lifetimeMap: Dict[str, Lifetime]) -> Dict[str, List[str]]:
        interferenceGraph: Dict[str, List[str]] = {}
        for name, lifetime in lifetimeMap.items():
            neighbors: List[str] = []
            for neighborName, neighborLifetime in lifetimeMap.items():
                if neighborName == name:
                    continue
                if lifetime.overlaps(neighborLifetime):
                    neighbors.append(neighborName)
            interferenceGraph[name] = neighbors
        return interferenceGraph

    def _calculateLifetimes(self, ctxt: NetworkContext, patternMemoryConstraint: PatternMemoryConstraint,
                            memoryLevel: str) -> Tuple[Dict[str, Lifetime], Dict[str, TensorMemoryConstraint]]:

        def hasLifetime(buffer: VariableBuffer) -> bool:
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
        lifetimeMap: Dict[str, Lifetime] = {}

        for stepIdx, nodeConstraint in enumerate(patternMemoryConstraint.nodeConstraints):
            for tensorName, tensorMemoryConstraint in nodeConstraint.tensorMemoryConstraints.items():
                if memoryLevel not in tensorMemoryConstraint.memoryConstraints:
                    continue

                buffer = ctxt.lookup(tensorName)
                assert isinstance(buffer, VariableBuffer)

                if not hasLifetime(buffer):
                    continue

                if tensorName in lifetimeMap:
                    lifetimeMap[tensorName].setEnd(stepIdx)
                else:
                    lifetimeMap[tensorName] = Lifetime(start = stepIdx, duration = 0)
                    tensorMap[tensorName] = tensorMemoryConstraint

                # LMACAN: Update end of lifetime for all visited aliases
                for name in lifetimeMap.keys():
                    if name != tensorName and ctxt.isAliased(tensorName, name):
                        lifetimeMap[name].setEnd(stepIdx)

        # JUNGVI: Align the lifetime of I/O tensors
        for tensorName, lifetime in lifetimeMap.items():
            buffer = ctxt.lookup(tensorName)
            assert isinstance(buffer, VariableBuffer)

            # Inputs should be alive from the beginning
            if buffer.is_input:
                lifetimeMap[tensorName] = Lifetime(start = 0, duration = lifetime.end)

            # Outputs should be alive until the end
            if buffer.is_output:
                lifetime.setEnd(len(patternMemoryConstraint.nodeConstraints))

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
    def _dealiasLifetimeMap(self, ctxt: NetworkContext, lifetimeMap: Dict[str, Lifetime]) -> Dict[str, Lifetime]:
        lifetimeMap = lifetimeMap.copy()
        if not self.tileScheduler:
            for name, lifetime in lifetimeMap.items():
                origin = ctxt.dealiasBuffer(name)

                if origin == name:
                    continue

                if ctxt.is_global(origin):
                    lifetimeMap[name] = Lifetime(0, lifetime.end)
                    continue

                originLifetime = lifetimeMap[origin]
                lifetimeMap[origin] = Lifetime(originLifetime.start, max(originLifetime.duration, lifetime.duration))
        return lifetimeMap

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
                                   patternMemoryConstraints: List[PatternMemoryConstraint],
                                   memoryHierarchy: MemoryHierarchy,
                                   memoryAllocStrategy: Literal["TetrisRandom", "TetrisCo-Opt"],
                                   memoryLevel: str = "L1"):

        if memoryLevel not in self.memoryMap:
            self.memoryMap[memoryLevel] = []

        for patternIdx, patternMemoryConstraint in enumerate(patternMemoryConstraints):
            lifetimeMap, tensorMap = self._calculateLifetimes(ctxt, patternMemoryConstraint, memoryLevel)

            #missingTensors = [
            #    tensorMc.tensorName for nodeConstr in patternMemoryConstraint.nodeConstraints
            #    for tensorMc in nodeConstr.tensorMemoryConstraints.values()
            #    if ctxt.lookup(tensorMc.tensorName)._deploy and tensorMc.tensorName not in tensorMap
            #]
            #assert len(missingTensors) == 0, f"Some tensors have not been assigned their memory constraint: {missingTensors}"

            lifetimeMap = self._dealiasLifetimeMap(ctxt, lifetimeMap)

            interferenceGraph = self._buildInterferenceGraph(lifetimeMap)

            numVars = len(interferenceGraph)

            adjacencyMatrix = self._buildAdjacencyMatrix(interferenceGraph, tensorMap)
            costVector = self._buildCostVector(ctxt, interferenceGraph, tensorMap, memoryLevel)

            # offset lifetimes by patternIdx
            for lt in lifetimeMap.values():
                lt.start += patternIdx

            blocks = [MemoryBlock(tensor, memoryLevel, lifetimeMap[tensor]) for tensor in interferenceGraph.keys()]

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
                                  allMemoryConstraints: List[PatternMemoryConstraint],
                                  memoryHierarchy: MemoryHierarchy,
                                  memoryAllocStrategy: Literal["TetrisRandom", "TetrisCo-Opt"],
                                  memoryLevel: str = "L1"):

        self.stringSuffix = self._stringSuffix + f"_{memoryLevel}"
        return self._scheduleMemoryConstraints(tilerModel, ctxt, allMemoryConstraints, memoryHierarchy,
                                               memoryAllocStrategy, memoryLevel)

    def constraintTileBuffersWithOverlappingLifetime(self, tilerModel: TilerModel, ctxt: NetworkContext,
                                                     patternMemoryConstraint: PatternMemoryConstraint,
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
            if len(permMatrix) == 0:
                return []

            if len(permMatrix) == 1:
                return [0]

            return permMatrix.nonzero()[1].tolist()

        for memoryLevel, patternList in self.memoryMap.items():
            for patternIdx, pattern in enumerate(patternList):

                permutationMatrix = self._permutationState[memoryLevel + f"_{patternIdx}"]

                if not isinstance(permutationMatrix, np.ndarray):
                    _permutationMatrix = self.getPMatrix(tilerModel, patternIdx, memoryLevel)
                else:
                    _permutationMatrix = permutationMatrix

                permList = permMatrix2permList(_permutationMatrix)
                permPattern = _permute(pattern, permList)

                blockNames = [block.name for block in permPattern]

                aliasedBlocks = []
                allocatedBlocks = []
                for blockIdx, block in enumerate(permPattern):
                    buffer = ctxt.lookup(block.name)
                    assert isinstance(buffer, VariableBuffer)

                    # SCHEREMO: If we're handling an active alias to a global buffer in their home memory level, we don't need to resolve addresses
                    origin = ctxt.dealiasBuffer(buffer.name)
                    if buffer.isAlias() and ctxt.is_global(origin) and buffer._memoryLevel == memoryLevel:
                        continue

                    # SCHEREMO: Don't fully unroll aliases here - this is pattern-sensitive!
                    if buffer.isAlias() and buffer.aliasedBuffer in blockNames:
                        aliasedBlocks.append((block, buffer.aliasedBuffer))
                        continue

                    upperEndVar = tilerModel.getVariable(
                        f"{self._COSTVARIABLENAME}_{blockIdx}{self._stringSuffix}_{memoryLevel}", patternIdx)
                    upperEnd = tilerModel._resolveVariable(upperEndVar)

                    base = 0
                    for other in allocatedBlocks:
                        if block.lifetime.overlaps(other.lifetime):
                            base = max(base, other.addrSpace.end)

                    block.addrSpace = AddressSpace(base = base, size = upperEnd - base)
                    allocatedBlocks.append(block)

                for block, alias in aliasedBlocks:
                    for refBlock in sorted(permPattern, key = lambda x: x.lifetime.start):
                        if refBlock.name == alias:
                            block.addrSpace = refBlock.addrSpace
                            break
