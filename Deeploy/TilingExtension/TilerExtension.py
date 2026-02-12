# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

# Create Monad that take a Deployer and make it TilerAware
# Define Tiler Obj centralize all tilling related functionalities for a given deployer.
# Like Template-T-Obj mapping, propagate cst, graph edition, etc

import copy
import csv
import os
import subprocess
from typing import Dict, List, Literal, Optional, OrderedDict, Tuple, Type, Union

import numpy as np
import onnx_graphsurgeon as gs
import plotly.graph_objects as go
import plotly.io as pio
from ortools.constraint_solver.pywrapcp import IntVar, SolutionCollector

import Deeploy.CommonExtensions.DataTypes as BasicDataTypes
from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.NetworkDeployers.NetworkDeployerWrapper import NetworkDeployerWrapper
from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, NodeBinding, NodeTemplate, ONNXLayer, Schedule, \
    SubGraph, TransientBuffer, VariableBuffer
from Deeploy.Logging import DEFAULT_LOGGER as log
from Deeploy.Logging import SUCCESS_MARK
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.MemoryLevelExtension.NetworkDeployers.MemoryLevelDeployer import MemoryDeployerWrapper, \
    MemoryLevelAwareDeployer, MemoryPlatform, MemoryPlatformWrapper, TargetMemoryLevelMapping
from Deeploy.TilingExtension.GenericFlow import GenericFlowState
from Deeploy.TilingExtension.MemoryConstraintFlows import GraphMemoryConstraintFlow, TensorMemLevelTuple, \
    convertFlowState2NodeMemoryConstraint
from Deeploy.TilingExtension.MemoryConstraints import MemoryConstraint, NodeMemoryConstraint, PatternMemoryConstraint, \
    TensorMemoryConstraint
from Deeploy.TilingExtension.MemoryScheduler import MemoryScheduler
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingTypes import AddressSpace, MemoryBlock

TilingSolution = List[PatternMemoryConstraint]
MemoryMap = Dict[str, List[List[MemoryBlock]]]

emptyTemplate = NodeTemplate("")


class Tiler():

    arenaName = "MEMORYARENA"
    memorySchedulerClass: Type[MemoryScheduler] = MemoryScheduler

    _MINIMALLOC_INPUT_FILENAME = "input_minimalloc"
    _MINIMALLOC_OUTPUT_FILENAME = "output_minimalloc"

    # Initialize with the list of TemplateTCFbinding
    def __init__(self, memoryHierarchy: MemoryHierarchy):

        self.memoryHierarchy = memoryHierarchy
        self.tilerModel: Optional[TilerModel] = None
        self.innerMemoryScheduler = self.memorySchedulerClass("_inner", tileScheduler = True)
        self.outerMemoryScheduler = self.memorySchedulerClass("_outer", tileScheduler = False)
        self.symbolicMemoryConstraints: Optional[List[PatternMemoryConstraint]] = None

        self._worstCaseBufferSize: Dict[str, int] = {}

        self.visualizeMemoryAlloc: bool = False
        self.memoryAllocStrategy: Literal["TetrisRandom", "TetrisCo-Opt", "MiniMalloc"] = "TetrisRandom"
        self.searchStrategy: Literal["min", "max", "random-max"] = "random-max"

    @property
    def worstCaseBufferSize(self):
        return self._worstCaseBufferSize

    def plotMemoryAlloc(self, memoryMap: Dict[str, List[List[MemoryBlock]]], ctxt: NetworkContext, deeployStateDir: str,
                        memoryHierarchy: MemoryHierarchy):

        os.makedirs(os.path.abspath(deeployStateDir), exist_ok = True)
        memoryAllocPlotPath = os.path.abspath(os.path.join(deeployStateDir, f"memory_alloc.html"))

        addTraceConfig = {"fill": "toself", "hoverinfo": "text", "mode": "lines", "line": dict(width = 2)}

        def plotSingleMemoryLevel(memoryLevel: MemoryLevel):
            """ Generates a single Plotly subplot for a memory level. """
            fig = go.Figure()
            constantBuffersOffset = 0

            infiniteLifetimeBuffers = [
                buffer for buffer in ctxt.globalObjects.values()
                if not self.arenaName in buffer.name and isinstance(buffer, ConstantBuffer)
            ]

            constantBuffersOffset = 0
            for ioBuffer in infiniteLifetimeBuffers:
                if not ioBuffer._memoryLevel == memoryLevel.name:
                    continue
                _ioSize = np.prod(ioBuffer.shape) * ioBuffer._type.referencedType.typeWidth // 8
                _maxLifetime = len(memoryMap[memoryLevel.name])
                fig.add_trace(
                    go.Scatter(x = [-0.5, -0.5, _maxLifetime + 0.5, _maxLifetime + 0.5],
                               y = [
                                   constantBuffersOffset, constantBuffersOffset + _ioSize,
                                   constantBuffersOffset + _ioSize, constantBuffersOffset
                               ],
                               name = ioBuffer.name,
                               text = ioBuffer.name,
                               **addTraceConfig))
                constantBuffersOffset += _ioSize

            for memoryMapStep in memoryMap[memoryLevel.name]:
                for buffer in memoryMapStep:
                    if buffer.addrSpace is None:
                        log.warning(
                            f"Buffer {buffer.name} has no address space assigned, skipping it in the memory allocation plot."
                        )
                        continue

                    fig.add_trace(
                        go.Scatter(x = [
                            buffer.lifetime.start - 0.5, buffer.lifetime.start - 0.5, buffer.lifetime.end + 0.5,
                            buffer.lifetime.end + 0.5
                        ],
                                   y = [
                                       constantBuffersOffset + buffer.addrSpace.base,
                                       constantBuffersOffset + buffer.addrSpace.end,
                                       constantBuffersOffset + buffer.addrSpace.end,
                                       constantBuffersOffset + buffer.addrSpace.base
                                   ],
                                   name = buffer.name,
                                   text = buffer.name,
                                   **addTraceConfig))

            fig.update_xaxes(title_text = "Lifetime")
            fig.update_yaxes(title_text = "Address Space (Bytes)")
            fig.update_layout(title = f"Memory Allocation - {memoryLevel.name}", showlegend = False)

            fig.add_trace(
                go.Scatter(
                    x = [-0.5, len(memoryMap[memoryLevel.name]) - 1.5],
                    y = [memoryLevel.size, memoryLevel.size],
                    name = f"{memoryLevel.name} Memory Size",
                    text = f"{memoryLevel.name} Memory Size",
                    line = dict(color = "red", width = 2, dash = "dash"),
                    fill = "toself",
                    hoverinfo = "text",
                    mode = "lines",
                ))

            return fig

        from Deeploy.TilingExtension.HtmlTemplates import getHtmlMemoryAllocationVisualisation, getSubplotHtml

        subplotHtml = ""
        for memoryLevelName in memoryMap.keys():
            figJson = pio.to_json(plotSingleMemoryLevel(memoryHierarchy.memoryLevels[memoryLevelName]))
            subplotHtml += getSubplotHtml(figJson, memoryLevelName)

        outputHtml = getHtmlMemoryAllocationVisualisation(subplotHtml)

        with open(memoryAllocPlotPath, "w", encoding = "utf-8") as f:
            f.write(outputHtml)

    def _convertCtxtToStaticSchedule(self, ctxt: NetworkContext,
                                     memoryMap: Dict[str, List[List[MemoryBlock]]]) -> NetworkContext:

        maxAddr: Dict[str, int] = {}

        for memory, patternList in memoryMap.items():
            currentMax = 0
            for blocks in patternList:
                blockNames = [block.name for block in blocks]
                for block in blocks:
                    buffer = ctxt.lookup(block.name)
                    assert isinstance(buffer, VariableBuffer)
                    # SCHEREMO: If alias buffers have zero cost, they don't contribute to the currentMax and their addrSpace is None
                    # TODO: This might need some way to check that we are looking at an output tensor
                    if buffer.isAlias() and (ctxt.is_global(ctxt.dealiasBuffer(buffer.name))
                                             or buffer.aliasedBuffer in blockNames):
                        continue

                    assert block.addrSpace is not None, \
                            f"Block {block.name} in the {memory} memory doesn't have an allocated address space even though it's neither aliasing a global buffer nor a neighboring buffer"
                    currentMax = max(currentMax, block.addrSpace.end)

            maxAddr[memory] = currentMax
            self._worstCaseBufferSize[memory] = currentMax

        arenas: Dict[str, VariableBuffer] = {}
        for memory, end in maxAddr.items():
            if end == 0:
                continue

            arenaName = f"{self.arenaName}_{memory}"

            arena = ctxt.VariableBuffer(arenaName, shape = [end])
            arena._type = PointerClass(BasicDataTypes.int8_t)
            ctxt.add(arena, "global")
            arena._instance = arena._type(arenaName, ctxt)
            arena._memoryLevel = memory

            # JUNGVI: Memory Arena buffers should be allocated first since other variable global buffers may belong to a memory arena
            # TODO: Check if this is still needed since now we are sorting the allocations in the memory codegen
            ctxt.globalObjects.move_to_end(arena.name, last = False)

            arenas[memory] = arena

        # SCHEREMO: Adapt homelevel tensors to their respective arena
        for memory, arena in arenas.items():
            patternList = memoryMap[memory]
            for blocks in patternList:
                blockNames = {node.name: node for node in blocks}
                for block in blocks:
                    buffer = ctxt.lookup(block.name)
                    assert isinstance(buffer, VariableBuffer)

                    if buffer._memoryLevel != memory:
                        continue

                    origin = ctxt.dealiasBuffer(buffer.name)

                    # Skip if the buffer is an alias to a global buffer or if the buffer itself is global
                    if ctxt.is_global(origin):
                        continue

                    if buffer.isAlias():
                        assert origin in blockNames, "I don't know what happens if an alias buffer resides in a different memory level then it's origin"
                        block = blockNames[origin]

                    assert block.addrSpace is not None, f"Expected block {block.name} to have an allocated address space."
                    buffer.allocTemplate = NodeTemplate("${name} = (${type.typeName}) " +
                                                        f"((char*){str(arena._instance)} + {block.addrSpace.base});")
                    buffer.deallocTemplate = emptyTemplate

        return ctxt

    def minimalloc(self, blocks: List[MemoryBlock], blockSizes: Dict[str, int], capacity: int, memory: str) -> None:
        # Write minimalloc input file
        with open(f"{self._MINIMALLOC_INPUT_FILENAME}.csv", mode = "w", newline = "") as file:
            writer = csv.writer(file, lineterminator = "\n")
            writer.writerow(["id", "lower", "upper", "size"])
            for block in blocks:
                writer.writerow(
                    [block.name,
                     str(block.lifetime.start),
                     str(block.lifetime.end + 1),
                     str(blockSizes[block.name])])

        # Check for the environment variable
        try:
            minimallocInstallDir = os.environ["MINIMALLOC_INSTALL_DIR"]
        except KeyError:
            raise KeyError("MINIMALLOC_INSTALL_DIR symbol not found!")

        # Run minimalloc
        minimallocOutput = subprocess.run([
            f"{minimallocInstallDir}/minimalloc", f"--capacity={capacity}",
            f"--input={self._MINIMALLOC_INPUT_FILENAME}.csv", f"--output={self._MINIMALLOC_OUTPUT_FILENAME}.csv"
        ],
                                          capture_output = True,
                                          text = True)

        # Check return code
        if minimallocOutput.returncode != 0:
            log.error(
                f"Memory allocator failed with return code {minimallocOutput.returncode} at memory level {memory} with capacity of {capacity} bytes!"
            )
            raise subprocess.CalledProcessError(minimallocOutput.returncode, " ".join(minimallocOutput.args))

        # Read minimalloc output into a dict
        allocations: Dict[str, Tuple[int, int]] = {}
        with open(f"{self._MINIMALLOC_OUTPUT_FILENAME}.csv", mode = "r", newline = "") as file:
            reader = csv.reader(file)
            _ = next(reader)  # Skip header
            for row in reader:
                _id, _, _, size, offset = row
                allocations[_id] = (int(size), int(offset))

        # Annotate blocks
        for block in blocks:
            size, offset = allocations[block.name]
            block.addrSpace = AddressSpace(base = offset, size = size)

    def computeTilingSchedule(self, ctxt: NetworkContext) -> TilingSolution:
        assert self.tilerModel is not None and self.symbolicMemoryConstraints is not None, "Set up the model before trying to compute a schedule!"
        collector = self.tilerModel.trySolveModel()
        tilingSolution = self._getTilingSolution(self.tilerModel, ctxt, collector, self.symbolicMemoryConstraints)
        if not self.memoryAllocStrategy == "MiniMalloc":
            assert self.tilerModel is not None
            log.debug(" - Extract Memory Allocation")
            self.innerMemoryScheduler.annotateSolution(ctxt, self.tilerModel)
            self.outerMemoryScheduler.annotateSolution(ctxt, self.tilerModel)
        return tilingSolution

    def computeMemoryMap(self, ctxt: NetworkContext, tilingSolution: TilingSolution) -> MemoryMap:
        memoryMap = {}

        for key in self.innerMemoryScheduler.memoryMap.keys():
            memoryMap[key] = [*self.innerMemoryScheduler.memoryMap[key], *self.outerMemoryScheduler.memoryMap[key]]

        if self.memoryAllocStrategy == "MiniMalloc":
            log.debug(" - Solve Memory Allocation with MiniMalloc")
            for memory, patterns in memoryMap.items():
                constantTensorOffset = self.outerMemoryScheduler.getConstantTensorOffset(ctxt, memory)
                capacity = self.memoryHierarchy.memoryLevels[memory].size - constantTensorOffset

                if memory == self.memoryHierarchy._defaultMemoryLevel.name:
                    blocks = patterns[-1]  # TODO: Investigate this -1
                    blockSizes = {block.name: ctxt.lookup(block.name).sizeInBytes() for block in blocks}
                    self.minimalloc(blocks, blockSizes, capacity, memory)
                else:
                    for idx, blocks in enumerate(patterns):
                        if len(blocks) == 0:
                            continue
                        nodeConstr = tilingSolution[idx].nodeConstraints[0]
                        blockSizes = {}
                        for block in blocks:
                            constr = nodeConstr.tensorMemoryConstraints[block.name].memoryConstraints[memory]
                            buff = ctxt.lookup(block.name)
                            assert isinstance(buff, VariableBuffer)
                            if isinstance(buff, TransientBuffer):
                                blockSizes[block.name] = constr.size
                            else:
                                blockSizes[block.name] = (constr.size * constr.multiBufferCoefficient *
                                                          buff._type.referencedType.typeWidth) // 8
                        self.minimalloc(blocks, blockSizes, capacity, memory)

            log.info(f" {SUCCESS_MARK} Memory allocation successful!")

        return memoryMap

    def annotateMemoryLevel(self, ctxt: NetworkContext, tilingSolution: TilingSolution,
                            memoryMap: Dict) -> NetworkContext:
        for idx, pattern in enumerate(tilingSolution):
            for nodeIdx, nodeConstraint in enumerate(pattern.nodeConstraints):
                for tensorConstraint in nodeConstraint.tensorMemoryConstraints.values():
                    for memoryConstraint in tensorConstraint.memoryConstraints.values():
                        patternList = memoryMap[memoryConstraint.memoryLevel]
                        blockPattern = patternList[idx]

                        # SCHEREMO: Don't try to annotate home base of tensor
                        if ctxt.lookup(tensorConstraint.tensorName
                                      )._memoryLevel == memoryConstraint.memoryLevel and not isinstance(
                                          ctxt.lookup(tensorConstraint.tensorName), TransientBuffer):
                            continue

                        _block = [memBlock for memBlock in blockPattern if memBlock.name == tensorConstraint.tensorName]

                        assert len(
                            _block
                        ) == 1, f"Missing or superfluous memory block {tensorConstraint.tensorName} allocation found in {_block}!"

                        block = _block[0]
                        memoryConstraint.addrSpace = block.addrSpace
        return ctxt

    def setupModel(self, ctxt: NetworkContext, schedule: Schedule, layerBinding: OrderedDict[str, ONNXLayer],
                   targetMemoryLevelMapping: TargetMemoryLevelMapping) -> NetworkContext:

        wrapSchedule: List[SubGraph] = []
        for entry in schedule:
            if isinstance(entry, gs.Node):
                wrapSchedule.append([entry])
            else:
                wrapSchedule.append(entry)

        tilerModel = TilerModel(searchStrategy = self.searchStrategy)
        tilerModel = self._setupGeometricConstraints(tilerModel, ctxt, wrapSchedule, layerBinding)
        tilerModel = self._setupTensorDimensionProducts(tilerModel, ctxt, wrapSchedule)
        tilerModel = self._setupHeuristics(tilerModel, ctxt, wrapSchedule)
        tilerModel, allSymbolicMemoryConstraints = self._setupMemoryConstraints(tilerModel, ctxt, wrapSchedule,
                                                                                layerBinding, targetMemoryLevelMapping)

        self.tilerModel = tilerModel
        self.symbolicMemoryConstraints = allSymbolicMemoryConstraints

        return ctxt

    # SCHEREMO: Return a integer factor or IntVar variable for the multi Buffer coefficient given the tiling path, hop and tensorName.
    def multiBufferStrategy(self, tilerModel: TilerModel, ctxt: NetworkContext, pattern: SubGraph, path: List[str],
                            hop: str, tensorName: str) -> Union[int, IntVar]:

        varBuffer = ctxt.lookup(tensorName)

        generalCoeff = 2

        if isinstance(varBuffer, TransientBuffer):
            coefficient = 1
        elif isinstance(varBuffer, ConstantBuffer):
            coefficient = generalCoeff
        else:
            coefficient = generalCoeff

        # if tensorName == pattern[-1].outputs[0].name:
        #     maxVal = (np.prod(varBuffer.shape) // (coefficient)).item()
        #     numElt = tilerModel.getTensorNumberOfEltVar(tensorName)
        #     constr = numElt <= maxVal

        #     if (constr != True):
        #         tilerModel.addConstraint(constr)

        return coefficient

    # SCHEREMO: Given a PatternMemoryConstraints object, propagate the IOBuffer freeing strategy.
    # Input: Single-buffered Liveness analysis of the input/output IO buffers that should be tiled
    # Output: Buffering-strategy aware liveness analysis of the input/output IO buffers

    # This version implements "static n-ple buffering"

    def propagateIOBufferStrategy(self, tileConstraintPattern: PatternMemoryConstraint, pattern: SubGraph,
                                  ctxt: NetworkContext) -> PatternMemoryConstraint:

        borderTensorStep = NodeMemoryConstraint()
        for patternStep in tileConstraintPattern.nodeConstraints:
            borderTensorStep += patternStep

        for idx in range(len(tileConstraintPattern.nodeConstraints)):
            tileConstraintPattern.nodeConstraints[idx] += borderTensorStep

        return tileConstraintPattern

    def _resolveTensorMemoryConstraint(self, tilerModel: TilerModel, ctxt: NetworkContext, collector: SolutionCollector,
                                       tensorConstraint: TensorMemoryConstraint) -> TensorMemoryConstraint:
        assert self.tilerModel is not None, "Can't resolve tensor memory constraints, tilerModel is None!"

        tensorName = tensorConstraint.tensorName
        solvedTensorConstraint = TensorMemoryConstraint(tensorName, {}, ctxt)

        for memoryLevel, memoryConstraint in tensorConstraint.memoryConstraints.items():
            size = self.tilerModel._resolveVariable(memoryConstraint.size)

            newMemoryConstraint: MemoryConstraint = MemoryConstraint(memoryLevel, size)
            multiBufferCoefficient = self.tilerModel._resolveVariable(memoryConstraint.multiBufferCoefficient)
            newMemoryConstraint.multiBufferCoefficient = multiBufferCoefficient

            if not isinstance(ctxt.lookup(tensorName), TransientBuffer):

                tensorShapeLen = len(ctxt.lookup(tensorName).shape)
                newShape: List[int] = []

                if isinstance(memoryConstraint.size, int):
                    newShape = ctxt.lookup(tensorName).shape
                else:
                    _, copyIdx = tilerModel.getNameCopyIdx(memoryConstraint.size.Name())
                    for i in range(tensorShapeLen):
                        newShape.append(
                            self.tilerModel._resolveVariable(tilerModel.getTensorDimVar(tensorName, i, copyIdx)))

                newMemoryConstraint.shape = tuple(newShape)

            solvedTensorConstraint.addMemoryConstraint(newMemoryConstraint)

        return solvedTensorConstraint

    def _getTilingSolution(self, tilerModel: TilerModel, ctxt: NetworkContext, collector: SolutionCollector,
                           allConstraints: List[PatternMemoryConstraint]) -> List[PatternMemoryConstraint]:

        retList = []

        def _checkResolve(ctxt, tensorName, tensorConstraint):

            if ctxt.is_global(tensorName) and len(tensorConstraint.memoryConstraints.values()) <= 1:
                return False
            if len(tensorConstraint.memoryConstraints.values()) <= 1 and not isinstance(
                    ctxt.lookup(tensorName), TransientBuffer):
                return False
            return True

        for patternConstraints in allConstraints:
            newMemoryConstraint = PatternMemoryConstraint()
            for stepConstraints in patternConstraints.nodeConstraints:
                newStepMemoryConstraint = NodeMemoryConstraint()
                for tensorName, tensorConstraint in stepConstraints.tensorMemoryConstraints.items():
                    if _checkResolve(ctxt, tensorName, tensorConstraint):
                        solvedTensorConstraint = self._resolveTensorMemoryConstraint(
                            tilerModel, ctxt, collector, tensorConstraint)
                        ioDir = stepConstraints.getIO(tensorName)
                        newStepMemoryConstraint.addTensorConstraint(solvedTensorConstraint, ioDir)

                newMemoryConstraint.addConstraint(newStepMemoryConstraint)
            retList.append(newMemoryConstraint)

        return retList

    def _setupTensorDimensionProducts(self, tilerModel: TilerModel, ctxt: NetworkContext,
                                      schedule: List[SubGraph]) -> TilerModel:

        for idx, pattern in enumerate(schedule):
            subGraph = gs.Graph(nodes = pattern)
            subgraphTensors: OrderedDict[str, gs.Tensor] = subGraph.tensors(check_duplicates = True)

            for _, tensor in subgraphTensors.items():
                if not ctxt.lookup(tensor.name)._deploy:
                    continue

                tilerModel.addTensorNumOfEltToModel(ctxt, tensor.name, idx)

        return tilerModel

    def _setupGeometricConstraints(self, tilerModel: TilerModel, ctxt: NetworkContext, schedule: List[SubGraph],
                                   layerBinding: OrderedDict[str, ONNXLayer]) -> TilerModel:

        # SCHEREMO: Each pattern is a decoupled sub-problem w.r.t the geometric constraints.
        # We need to regenerate dimension variables for each tensor
        # This is done by setting the copyIdx in the tilerModel

        for idx, pattern in enumerate(schedule):
            tilerModel.copyIdx = idx

            for node in pattern:

                if node.name not in layerBinding.keys():
                    continue

                parseDict = layerBinding[node.name].mapper.parser.operatorRepresentation
                template = layerBinding[node.name].mapper.binder.template

                tilerModel = template.tileConstraint.addGeometricalConstraint(tilerModel,
                                                                              parseDict = parseDict,
                                                                              ctxt = ctxt)

                tilerModel = template.tileConstraint.addPolicyConstraint(tilerModel, parseDict = parseDict, ctxt = ctxt)

        return tilerModel

    def _setupHeuristics(self, tilerModel: TilerModel, ctxt: NetworkContext, schedule: List[SubGraph]) -> TilerModel:

        for idx, pattern in enumerate(schedule):

            patternTensorList = []
            seenTensorNameList = []
            for node in pattern:
                for gsTensor in node.inputs + node.outputs:
                    ctxtTensor = ctxt.lookup(gsTensor.name)
                    if ctxtTensor.name not in seenTensorNameList:
                        seenTensorNameList.append(ctxtTensor.name)
                        patternTensorList.append(ctxtTensor)

            patternMemSizeExpr: IntVar = 0
            for tensor in patternTensorList:
                if not ctxt.lookup(tensor.name)._deploy:
                    continue

                patternMemSizeExpr += tilerModel.getTensorNumberOfEltVar(
                    tensorName = tensor.name, copyIdx = idx) * (tensor._type.referencedType.typeWidth // 8)

            if isinstance(patternMemSizeExpr, int):
                _max = patternMemSizeExpr
            else:
                _max = patternMemSizeExpr.Max()

            patternVariable = tilerModel.addVariable(name = "DEEPLOY_PATTERN_MEM",
                                                     lowerBound = 1,
                                                     upperBound = _max,
                                                     copyIdx = idx)
            tilerModel.addConstraint(patternVariable == patternMemSizeExpr)

            tilerModel.addObjective(patternVariable, 'maximize')

        return tilerModel

    def _setupMemoryConstraints(
            self, tilerModel: TilerModel, ctxt: NetworkContext, schedule: List[SubGraph],
            layerBinding: OrderedDict[str, ONNXLayer],
            targetMemoryLevelMapping: TargetMemoryLevelMapping) -> Tuple[TilerModel, List[PatternMemoryConstraint]]:

        allMemoryConstraints = self._generateAllMemoryConstraints(tilerModel, ctxt, schedule, layerBinding,
                                                                  targetMemoryLevelMapping)

        outerMemoryConstraints = PatternMemoryConstraint()
        for constraint in allMemoryConstraints:
            for nodeConstraint in constraint.nodeConstraints:
                outerMemoryConstraints.addConstraint(nodeConstraint)

        if self.memoryAllocStrategy == "MiniMalloc":
            # JUNGVI: This method adds the memory constraints in case of decoupled tiling and memory allocation.
            self.outerMemoryScheduler.constraintTileBuffersWithOverlappingLifetime(tilerModel, ctxt,
                                                                                   outerMemoryConstraints,
                                                                                   self.memoryHierarchy)

        for level in self.memoryHierarchy.memoryLevels.keys():
            self.outerMemoryScheduler.scheduleMemoryConstraints(tilerModel, ctxt, [outerMemoryConstraints],
                                                                self.memoryHierarchy, self.memoryAllocStrategy, level)

        # Update inner memoryHierarchy with outer constraints
        innerMemoryHierarchy = MemoryHierarchy([])
        for level, memLevel in self.memoryHierarchy.memoryLevels.items():
            newMemLevel = copy.copy(memLevel)

            if not self.memoryAllocStrategy == "MiniMalloc":
                outerConstraint = tilerModel.getVariable(self.outerMemoryScheduler.getSymbolicCostName(0, level), 0)
                newMemLevel.size = newMemLevel.size - outerConstraint

            innerMemoryHierarchy._add(newMemLevel)

        for level in innerMemoryHierarchy.memoryLevels.keys():
            self.innerMemoryScheduler.scheduleMemoryConstraints(tilerModel, ctxt, allMemoryConstraints,
                                                                innerMemoryHierarchy, self.memoryAllocStrategy, level)

        return tilerModel, allMemoryConstraints

    def _generateAllMemoryConstraints(
            self, tilerModel: TilerModel, ctxt: NetworkContext, schedule: List[SubGraph],
            layerBinding: OrderedDict[str, ONNXLayer],
            targetMemoryLevelMapping: TargetMemoryLevelMapping) -> List[PatternMemoryConstraint]:

        dynamicTensorConstraints, constantTensorConstraints = self._generateMemoryConstraints(
            tilerModel, ctxt, schedule, layerBinding, targetMemoryLevelMapping)

        allConstraints: List[PatternMemoryConstraint] = []
        # Initialize structures

        for pattern in dynamicTensorConstraints:
            allPattern = PatternMemoryConstraint()
            for step in pattern.nodeConstraints:
                allStep = step + constantTensorConstraints
                allPattern.addConstraint(allStep)
            allConstraints.append(allPattern)

        return allConstraints

    def _generateMemoryConstraints(
        self, tilerModel: TilerModel, ctxt: NetworkContext, schedule: List[SubGraph],
        layerBinding: OrderedDict[str, ONNXLayer], targetMemoryLevelMapping: TargetMemoryLevelMapping
    ) -> Tuple[List[PatternMemoryConstraint], NodeMemoryConstraint]:

        # SCHEREMO: Construct non-double-buffered constraints of local variable buffers

        outerVariableConstraints, innerVariableConstraints = self._generateVariableBufferConstraints(
            tilerModel, ctxt, schedule, layerBinding, targetMemoryLevelMapping)

        # SCHEREMO: Construct global buffer constraints

        constantBufferConstraint = self._generateBufferConstraints(ctxt)

        # SCHEREMO: Construct first-level constraint set (all global buffers + tensors stored in higher level)

        firstLevelConstraints: List[PatternMemoryConstraint] = copy.copy(outerVariableConstraints)
        for patternConstraint in firstLevelConstraints:
            for idx in range(len(patternConstraint.nodeConstraints)):
                patternConstraint.nodeConstraints[idx] += constantBufferConstraint

        # SCHEREMO: Construct constraint set for tiled tensors (including double buffering, excluding static global constraints)
        tiledTensorConstraints: List[PatternMemoryConstraint] = self._generateTilePathConstraints(
            tilerModel, ctxt, firstLevelConstraints, innerVariableConstraints, schedule)

        # SCHEREMO: Construct constraint set for tiled tensors + local-only tensors (dynamic tensor set)
        dynamicTensorConstraints: List[PatternMemoryConstraint] = []
        for tilingConstraints, innerConstraints in zip(tiledTensorConstraints, innerVariableConstraints):

            dynamicTensorPattern = PatternMemoryConstraint()
            for tilingPatternStep, innerPatternStep in zip(tilingConstraints.nodeConstraints,
                                                           innerConstraints.nodeConstraints):
                dynamicTensorPatternStep = copy.copy(tilingPatternStep)

                # Pick all constraints that are purely internal
                for innerTensorName, innerTensor in innerPatternStep.tensorMemoryConstraints.items():
                    if not any([
                            innerTensorName == dynamicTensorName
                            for dynamicTensorName, tensor in dynamicTensorPatternStep.tensorMemoryConstraints.items()
                    ]):
                        ioDir = tilingPatternStep.getIO(innerTensorName)
                        dynamicTensorPatternStep.addTensorConstraint(innerTensor, ioDir)
                dynamicTensorPattern.addConstraint(dynamicTensorPatternStep)
            dynamicTensorConstraints.append(dynamicTensorPattern)

        # SCHEREMO: Construct unkilled tensor set
        inplaceTensorConstraints: List[PatternMemoryConstraint] = []
        for tilingConstraints, outerConstraints in zip(dynamicTensorConstraints, firstLevelConstraints):
            dynamicTensorPattern = PatternMemoryConstraint()
            for tilingPatternStep, outerPatternStep in zip(tilingConstraints.nodeConstraints,
                                                           outerConstraints.nodeConstraints):
                dynamicTensorPatternStep = copy.copy(tilingPatternStep)

                # Pick all constraints that are purely internal
                for outerTensorName, outerTensor in outerPatternStep.tensorMemoryConstraints.items():
                    if not any(
                        [(outerTensorName == dynamicTensorName) or (ctxt.is_global(outerTensorName))
                         for dynamicTensorName, tensor in dynamicTensorPatternStep.tensorMemoryConstraints.items()]):
                        dynamicTensorPatternStep.addTensorConstraint(outerTensor, "intermediate")
                dynamicTensorPattern.addConstraint(dynamicTensorPatternStep)
            inplaceTensorConstraints.append(dynamicTensorPattern)

        return inplaceTensorConstraints, constantBufferConstraint

    def _generateTilePath(self, tilerModel: TilerModel, ctxt: NetworkContext,
                          tensorMemoryConstraint: TensorMemoryConstraint, pattern: SubGraph) -> TensorMemoryConstraint:

        assert len(tensorMemoryConstraint.memoryConstraints.keys()
                  ) == 2, "Can't generate a tile path for more than one hierarchy level!"

        tensorName = tensorMemoryConstraint.tensorName

        valList = list(tensorMemoryConstraint.memoryConstraints.values())
        constraintA = valList[0]
        constraintB = valList[1]

        # SCHEREMO : Base is whichever constraint is constant
        base = constraintA if isinstance(constraintA.size, int) else constraintB
        end = constraintA if base == constraintB else constraintB

        path = self.memoryHierarchy.bfs(base.memoryLevel, end.memoryLevel)
        requiredHops = path[1:]

        returnTensorConstraint = TensorMemoryConstraint(tensorName, {}, ctxt)
        returnTensorConstraint.addMemoryConstraint(base)

        for hop in requiredHops:
            factor = self.multiBufferStrategy(tilerModel, ctxt, pattern, path, hop, tensorName)
            assert factor >= 1 and isinstance(factor, int), "Invalid factor!"

            memConstraint = MemoryConstraint(hop, end.size)
            memConstraint.multiBufferCoefficient = factor
            returnTensorConstraint.addMemoryConstraint(memConstraint)

        return returnTensorConstraint

    def _generateIntermediateTilingSteps(self, tilerModel: TilerModel, ctxt: NetworkContext,
                                         sourceStep: NodeMemoryConstraint, destinationStep: NodeMemoryConstraint,
                                         pattern: SubGraph) -> NodeMemoryConstraint:
        tileConstraintStep = NodeMemoryConstraint()

        mergedStep = sourceStep + destinationStep
        tileTensorConstraints = [
            tensor for tensor in mergedStep.tensorMemoryConstraints.values()
            if len(tensor.memoryConstraints.values()) > 1
        ]

        for tileTensor in tileTensorConstraints:
            tiledTensor = self._generateTilePath(tilerModel, ctxt, tileTensor, pattern)
            ioDir = mergedStep.getIO(tileTensor.tensorName)
            tileConstraintStep.addTensorConstraint(tiledTensor, ioDir)

        return tileConstraintStep

    def _generateTilePathConstraints(self, tilerModel: TilerModel, ctxt: NetworkContext,
                                     sourceConstraints: List[PatternMemoryConstraint],
                                     destinationConstraints: List[PatternMemoryConstraint],
                                     schedule: List[SubGraph]) -> List[PatternMemoryConstraint]:

        tileConstraints = []

        for idx, (sourceConstraint, destinationConstraint) in enumerate(zip(sourceConstraints, destinationConstraints)):

            tilerModel.copyIdx = idx

            assert (len(sourceConstraint.nodeConstraints) == 1
                   ), "source pattern must be constant and single step, since it's live throughout the pattern!"
            sourcePatternStep = sourceConstraint.nodeConstraints[0]

            tileConstraint = PatternMemoryConstraint()

            for destinationConstraintStep in destinationConstraint.nodeConstraints:
                tileConstraintStep = self._generateIntermediateTilingSteps(tilerModel, ctxt, sourcePatternStep,
                                                                           destinationConstraintStep, schedule[idx])
                tileConstraint.addConstraint(tileConstraintStep)

            propagatedTileConstraint = self.propagateIOBufferStrategy(tileConstraint, schedule[idx], ctxt)
            assert len(propagatedTileConstraint.nodeConstraints) == len(tileConstraint.nodeConstraints)

            tileConstraints.append(propagatedTileConstraint)

        return tileConstraints

    def _generateBufferConstraints(self, ctxt: NetworkContext) -> NodeMemoryConstraint:

        constantGlobalConstraint: NodeMemoryConstraint = NodeMemoryConstraint()
        constantGlobalBuffers = [
            node for node in ctxt.globalObjects.values() if isinstance(node, ConstantBuffer) and node._deploy == True
        ]

        for constantBuffer in constantGlobalBuffers:

            tensorName = constantBuffer.name

            memorySize = int(np.prod(ctxt.lookup(tensorName).shape))

            elementMemorySize = memorySize
            memoryConstraint = MemoryConstraint(constantBuffer._memoryLevel, elementMemorySize)
            tensorConstraint = TensorMemoryConstraint(constantBuffer.name,
                                                      {memoryConstraint.memoryLevel: memoryConstraint}, ctxt)
            constantGlobalConstraint.addTensorConstraint(tensorConstraint, "input")

        return constantGlobalConstraint

    def _generateVariableBufferConstraints(
        self, tilerModel: TilerModel, ctxt: NetworkContext, schedule: List[SubGraph],
        layerBinding: OrderedDict[str, ONNXLayer], targetMemoryLevelMapping: TargetMemoryLevelMapping
    ) -> Tuple[List[PatternMemoryConstraint], List[PatternMemoryConstraint]]:

        def deltaFlow(
                patternFlow: List[GenericFlowState[TensorMemLevelTuple]]) -> GenericFlowState[TensorMemLevelTuple]:

            initialFlow = patternFlow[0]
            endFlow = patternFlow[1]

            # SCHEREMO: The genset and killset of the innerflow are correct; however, since we now pass the initialliveset of the pattern to the constraint flow. we need to remove bypassed tensors
            mergedLiveSet = initialFlow.liveSet - endFlow.liveSet
            mergedGenSet = initialFlow.genSet
            mergedKillSet = initialFlow.killSet

            mergedFlow = GenericFlowState[TensorMemLevelTuple](mergedLiveSet, mergedKillSet, mergedGenSet)

            return mergedFlow

        initialLiveBuffers = {
            value.name
            for value in ctxt.globalObjects.values()
            if (isinstance(value, ctxt.VariableBuffer) and value._users != [])
        }

        producedBuffers = {layer.node.outputs[0].name for layer in layerBinding.values()}
        inputBufferNames = initialLiveBuffers - producedBuffers
        inputBuffers = [ctxt.lookup(name) for name in inputBufferNames]

        initialLiveTensors = {TensorMemLevelTuple(buf.name, buf._memoryLevel) for buf in inputBuffers}

        constraintFlow = GraphMemoryConstraintFlow(ctxt, targetMemoryLevelMapping)
        graphFlowStates = constraintFlow.flow(schedule, initialLiveTensors)

        innerMemConstraints: List[PatternMemoryConstraint] = []
        outerMemConstraints: List[PatternMemoryConstraint] = []

        for idx, pattern in enumerate(schedule):

            tilerModel.copyIdx = idx

            innerPatternMemoryConstraints = PatternMemoryConstraint()
            outerPatternMemoryConstraints = PatternMemoryConstraint()

            outerFlowState = graphFlowStates[idx]
            patternFlow = constraintFlow._patternFlowStates[idx]

            dynamicOuterBufferConstraints = convertFlowState2NodeMemoryConstraint(tilerModel,
                                                                                  ctxt,
                                                                                  outerFlowState,
                                                                                  useMax = True)

            outerPatternMemoryConstraints.addConstraint(dynamicOuterBufferConstraints)
            outerMemConstraints.append(outerPatternMemoryConstraints)

            mergedFlow = [deltaFlow(patternFlow)]

            for step, innerFlowState in zip(pattern, mergedFlow):
                transientBufferConstraints = self._generatePatternStepTransientBufferConstraints(
                    tilerModel, ctxt, layerBinding, step, targetMemoryLevelMapping)

                dynamicInnerBufferConstraints = convertFlowState2NodeMemoryConstraint(tilerModel,
                                                                                      ctxt,
                                                                                      innerFlowState,
                                                                                      useMax = False)

                innerPatternMemoryConstraints.addConstraint(transientBufferConstraints + dynamicInnerBufferConstraints)

            innerMemConstraints.append(innerPatternMemoryConstraints)

        return outerMemConstraints, innerMemConstraints

    def _generatePatternStepTransientBufferConstraints(
            self, tilerModel: TilerModel, ctxt: NetworkContext, layerBinding: OrderedDict[str, ONNXLayer],
            step: gs.Node, targetMemoryLevelMapping: TargetMemoryLevelMapping) -> NodeMemoryConstraint:

        patternStepTransientBufferSizes = NodeMemoryConstraint()

        template = layerBinding[step.name].mapper.binder.template

        symbolicNodeRep = template.tileConstraint.constructSymbolicNodeRep(
            tilerModel, parseDict = layerBinding[step.name].mapper.parser.operatorRepresentation, ctxt = ctxt)

        transientBufferList: List[Tuple[str,
                                        Union[int,
                                              IntVar]]] = template.computeTransientBuffersSize(ctxt, symbolicNodeRep)

        for tensorName, memorySize in transientBufferList:

            # SCHEREMO: Assume transientbuffers end up in the same level as their user's main input
            memoryLevelName = targetMemoryLevelMapping.lookup(step.name, step.inputs[0].name)
            ctxt.lookup(tensorName)._memoryLevel = memoryLevelName

            transientSize = tilerModel.addTransientBufferSizeToModel(tensorName, memorySize)

            #memoryLevelName = self.memoryHierarchy.getDefaultMemoryLevel().name

            transientMemoryConstraint = MemoryConstraint(memoryLevelName, transientSize)
            transientBufferConstraint = TensorMemoryConstraint(tensorName, {memoryLevelName: transientMemoryConstraint},
                                                               ctxt)
            patternStepTransientBufferSizes.addTensorConstraint(transientBufferConstraint, "intermediate")

        return patternStepTransientBufferSizes

    def assertLayerWiseTiling(self, schedule: List[List[gs.Node]]) -> bool:
        for pattern in schedule:
            if len(pattern) > 1:
                return False

        return True

    def assertUniformMemoryLevelAllocation(self, ctxt: NetworkContext, defaultMemoryLevel: str) -> bool:
        for buffer in ctxt.localObjects.values():
            if buffer._memoryLevel != defaultMemoryLevel:
                return False
        return True

    def testTilingSolutionCorrectness(self, tilingSolution: TilingSolution) -> None:
        # LMACAN: Assert buffer sizes are word aligned as per comment in MemoryScheduler.py:MemoryScheduler._buildCostVector()
        byteAlignment = MemoryScheduler.byteAlignment
        for patternMemoryConstraint in tilingSolution:
            for nodeMemoryConstraint in patternMemoryConstraint.nodeConstraints:
                for tensorMemoryConstraint in nodeMemoryConstraint.tensorMemoryConstraints.values():
                    for memoryConstraint in tensorMemoryConstraint.memoryConstraints.values():
                        if memoryConstraint.addrSpace is not None:
                            assert isinstance(memoryConstraint.multiBufferCoefficient, int)
                            bufferSize = memoryConstraint.addrSpace.size // memoryConstraint.multiBufferCoefficient
                            assert bufferSize % byteAlignment == 0, f"Buffer in {memoryConstraint} is not {byteAlignment} byte aligned"

    def testMemoryMapCorrectness(self, memoryMap: Dict[str, List[List[MemoryBlock]]], graph: gs.Graph,
                                 schedule: Schedule) -> None:

        memoryBlockMap = {
            memoryBlock.name: memoryBlock for levelMemoryMap in memoryMap.values() for memoryBlock in levelMemoryMap[-1]
        }

        # JUNGVI: Assert output buffers are alive until the end
        for tensor in graph.outputs:
            assert memoryBlockMap[tensor.name].lifetime.end == len(schedule), \
                    "Invalid memory map! Output buffer is not alive at the last step!"

        # JUNGVI: Assert input buffers are alive at the beginning
        for inputBuffer in graph.inputs:
            assert memoryBlockMap[inputBuffer.name].lifetime.start == 0, \
                    "Invalid memory map! Input buffer is not alive at step 0!"

        # JUNGVI: Assert that at every computation step, the required buffers are alive somewhere in memory
        for stepIdx, pattern in enumerate(schedule):
            node = pattern[0]
            nodeIO = [node for node in node.inputs + node.outputs if not isinstance(node, gs.Constant)]
            for tensor in nodeIO:
                lifetime = memoryBlockMap[tensor.name].lifetime
                assert lifetime.contains(stepIdx), \
                        f"Invalid memory map! Buffer {tensor.name} is not alive at step {stepIdx}!"


class TilerDeployerWrapper(NetworkDeployerWrapper):

    def __init__(self, deployer: Union[MemoryLevelAwareDeployer, MemoryDeployerWrapper], tilerCls: Type[Tiler] = Tiler):
        super().__init__(deployer)
        assert isinstance(self.Platform, (MemoryPlatform, MemoryPlatformWrapper)), \
            f"Platform should be a MemoryPlatform or MemoryPlatformWrapper! Got {type(self.Platform).__name__}"
        self.tiler = tilerCls(self.Platform.memoryHierarchy)

    @property
    def worstCaseBufferSize(self):
        return self.tiler.worstCaseBufferSize

    def tile(self, tilingSolution: Optional[TilingSolution] = None, memoryMap: Optional[MemoryMap] = None):
        assert (tilingSolution is None and memoryMap is None) or (tilingSolution is not None and memoryMap is not None), \
            "You need to provide both the manual tilingSolution and the memoryMap to override tiling."

        schedule = self.scheduler(self.graph)

        if tilingSolution is None and memoryMap is None:
            # JUNGVI: Currently using MiniMalloc is only supported for layer-wise execution and all tensors in the default memory level.
            if self.tiler.memoryAllocStrategy == "MiniMalloc":
                assert self.tiler.assertLayerWiseTiling(schedule), "Using MiniMalloc and DFT is not supported!"
                assert self.tiler.assertUniformMemoryLevelAllocation(
                    self.ctxt, self.Platform.memoryHierarchy._defaultMemoryLevel.name
                ), "All tensors have to be in the default memory level when using MiniMalloc!"

            log.debug(" - Setup Constraint Model")
            self.tiler.setupModel(ctxt = self.ctxt,
                                  schedule = schedule,
                                  layerBinding = self.layerBinding,
                                  targetMemoryLevelMapping = self.getTargetMemoryLevelMapping())
            tilingSolution = self.tiler.computeTilingSchedule(self.ctxt)

            memoryMap = self.tiler.computeMemoryMap(self.ctxt, tilingSolution)

        assert tilingSolution is not None and memoryMap is not None

        log.debug(" - Test Tiling Solution Correctness")
        self.tiler.testTilingSolutionCorrectness(tilingSolution)

        log.debug(" - Annotate Memory Levels")
        self.tiler.annotateMemoryLevel(self.ctxt, tilingSolution, memoryMap)

        self.ctxt = self.tiler._convertCtxtToStaticSchedule(self.ctxt, memoryMap)

        if self.tiler.visualizeMemoryAlloc:
            log.info(f" > Export Memory Allocation Visualization to {self.deeployStateDir}")
            self.tiler.plotMemoryAlloc(memoryMap, self.ctxt, self.deeployStateDir, self.Platform.memoryHierarchy)

        log.debug(" - Test Memory Map Correctness")
        self.tiler.testMemoryMapCorrectness(memoryMap, self.graph, schedule)

        # SCHEREMO: Annotate execution block with solution
        for layer, pattern in zip(self.layerBinding.values(), tilingSolution):
            layer.mapper.binder.executionBlock.patternMemoryConstraint = pattern

        # SCHEREMO: Code generation STUB

    def bind(self):
        if not super().bind():
            return False

        log.info("- Performing Tiling and Memory Allocation")
        self.tile()
        return True

    def _printMemorySummary(self):
        log.info("")
        log.info("Memory Usage Report:")
        log.info(f"  {'Level':<14} {'Capacity (bytes)':>10} {'Total':>10} (    Static + Dynamic   ) (Usage )")
        log.info("  " + "-" * 78)

        for level, dynamicSize in self.worstCaseBufferSize.items():
            staticSize = self.tiler.outerMemoryScheduler.getConstantTensorOffset(self.ctxt, level)
            capacity = self.tiler.memoryHierarchy.memoryLevels[level].size
            total = staticSize + dynamicSize

            log.info(f"  {level:<20} {capacity:10,} {total:10,d} "
                     f"({staticSize:10,d} + {dynamicSize:10,d}) "
                     f"({total / capacity * 100:5.1f}%)")


def TilingReadyNodeBindings(nodeBindings: List[NodeBinding], tileConstraint: TileConstraint) -> List[NodeBinding]:
    '''
    Apply the TillingReadyNodeTemplate to the template of each NodeBinding.
    '''
    nodeBindingsCopy = copy.deepcopy(nodeBindings)  #.copy()
    for binding in nodeBindingsCopy:
        binding.template.tileConstraint = tileConstraint

    return nodeBindingsCopy
