# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

# Create Monad that take a Deployer and make it TilerAware
# Define Tiler Obj centralize all tilling related functionalities for a given deployer.
# Like Template-T-Obj mapping, propagate cst, graph edition, etc

import copy
import csv
import math
import os
import subprocess
from collections import OrderedDict
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

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
from Deeploy.TilingExtension.MemoryConstraintFlows import InnerBufferLivenessAnalysis, OuterBufferLivenessAnalysis
from Deeploy.TilingExtension.MemoryConstraints import MemoryConstraint, NodeMemoryConstraint, \
    PatternMemoryConstraints, TensorMemoryConstraint
from Deeploy.TilingExtension.MemoryScheduler import MemoryBlock, MemoryScheduler
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import Objective, TilerModel

TilingSolution = List[PatternMemoryConstraints]
MemoryMap = Dict[str, List[List[MemoryBlock]]]

_deallocTemplate = NodeTemplate("")


class Tiler():

    arenaName = "MEMORYARENA"
    memorySchedulerClass: Type[MemoryScheduler] = MemoryScheduler

    _MINIMALLOC_INPUT_FILENAME = "input_minimalloc"
    _MINIMALLOC_OUTPUT_FILENAME = "output_minimalloc"

    # Initialize with the list of TemplateTCFbinding
    def __init__(self, memoryHierarchy: MemoryHierarchy, testName: Optional[str] = None, workDir: Optional[str] = None):

        self.memoryHierarchy = memoryHierarchy
        self.tilerModel: Optional[TilerModel] = None
        self.innerMemoryScheduler = self.memorySchedulerClass("_inner", tileScheduler = True)
        self.outerMemoryScheduler = self.memorySchedulerClass("_outer", tileScheduler = False)
        self.symbolicMemoryConstraints: Optional[List[PatternMemoryConstraints]] = None

        self._worstCaseBufferSize: Dict[str, int] = {}

        self.visualizeMemoryAlloc: bool = False
        self.memoryAllocStrategy: Literal["TetrisRandom", "TetrisCo-Opt", "MiniMalloc"] = "TetrisRandom"
        self.searchStrategy: Literal["min", "max", "random-max"] = "random-max"

        if workDir is not None:
            os.makedirs(workDir, exist_ok = True)
            minimalloc_base = os.path.join(workDir, self._MINIMALLOC_INPUT_FILENAME)
            minimalloc_output_base = os.path.join(workDir, self._MINIMALLOC_OUTPUT_FILENAME)
        else:
            minimalloc_base = self._MINIMALLOC_INPUT_FILENAME
            minimalloc_output_base = self._MINIMALLOC_OUTPUT_FILENAME

        if testName is not None:
            # VJUNG: Sanitize path
            safe_test_name = testName.replace("/", "_").replace("\\", "_")
            self._minimalloc_input = f"{minimalloc_base}_{safe_test_name}"
            self._minimalloc_output = f"{minimalloc_output_base}_{safe_test_name}"
        else:
            self._minimalloc_input = minimalloc_base
            self._minimalloc_output = minimalloc_output_base

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
                    if not hasattr(buffer, "_addrSpace") or buffer._addrSpace is None:
                        log.warning(
                            f"Buffer {buffer.name} has no address space assigned, skipping it in the memory allocation plot."
                        )
                        continue

                    fig.add_trace(
                        go.Scatter(x = [
                            buffer._lifetime[0] - 0.5, buffer._lifetime[0] - 0.5, buffer._lifetime[1] + 0.5,
                            buffer._lifetime[1] + 0.5
                        ],
                                   y = [
                                       constantBuffersOffset + buffer._addrSpace[0],
                                       constantBuffersOffset + buffer._addrSpace[1],
                                       constantBuffersOffset + buffer._addrSpace[1],
                                       constantBuffersOffset + buffer._addrSpace[0]
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

        for memoryLevel, patternList in memoryMap.items():
            currentMax = 0
            for nodeList in patternList:
                blockNames = [block.name for block in nodeList]
                for node in nodeList:

                    _buffer = ctxt.lookup(node.name)
                    # SCHEREMO: If alias buffers have zero cost, they don't contribute to the currentMax and their addrSpace is None
                    if hasattr(_buffer, "_alias") and (ctxt.is_global(_buffer._alias) or _buffer._alias in blockNames):
                        continue

                    currentMax = max(currentMax, node._addrSpace[1])

            maxAddr[memoryLevel] = currentMax
            self._worstCaseBufferSize[memoryLevel] = currentMax

        for level, addrSpace in maxAddr.items():
            if addrSpace == 0:
                continue

            arenaName = f"{self.arenaName}_{level}"

            scratchBuffer = ctxt.VariableBuffer(arenaName, [addrSpace])
            scratchBuffer._type = PointerClass(BasicDataTypes.int8_t)
            ctxt.add(scratchBuffer, "global")
            scratchBuffer._instance = scratchBuffer._type(arenaName, ctxt)
            scratchBuffer._memoryLevel = level

            # JUNGVI: Memory Arena buffers should be allocated first since other variable global buffers may belong to a memory arena
            ctxt.globalObjects.move_to_end(scratchBuffer.name, last = False)

        # SCHEREMO: Adapt homelevel tensors to their respective arena
        for memoryLevel, patternList in memoryMap.items():
            if not ctxt.is_global(f"{self.arenaName}_{memoryLevel}"):
                continue
            staticBuf = ctxt.lookup(f"{self.arenaName}_{memoryLevel}")
            for nodeList in patternList:
                blockNames = [block.name for block in nodeList]
                for node in nodeList:
                    tensorName = node.name
                    _buffer = ctxt.lookup(tensorName)

                    if _buffer._memoryLevel != memoryLevel:
                        continue

                    if hasattr(_buffer, "_alias") and ctxt.is_global(_buffer._alias):
                        continue

                    if hasattr(_buffer, "_alias") and _buffer._alias in blockNames:

                        alias = ctxt.dealiasBuffer(tensorName)
                        aliasNodes = [node for node in nodeList if node.name == alias]

                        assert len(aliasNodes) == 1, f"alias {alias} references more than one node!"

                        aliasNode = aliasNodes[0]

                        _buffer.allocTemplate = NodeTemplate(
                            " \
                        ${name} = (${type.typeName}) " +
                            f"((char*){str(staticBuf._instance)} + {aliasNode.addrSpace[0]});")
                        _buffer.deallocTemplate = _deallocTemplate

                        continue

                    offset = node.addrSpace[0]

                    _buffer.allocTemplate = NodeTemplate(" \
                    ${name} = (${type.typeName}) " + f"((char*){str(staticBuf._instance)} + {offset});")
                    _buffer.deallocTemplate = _deallocTemplate

        return ctxt

    def minimalloc(self, memoryMap, ctxt, nodeMemoryConstraint, capacity: int, memoryLevel: str):

        with open(f"{self._minimalloc_input}.csv", mode = "w", newline = "") as file:
            writer = csv.writer(file, lineterminator = "\n")
            writer.writerow(["id", "lower", "upper", "size"])
            for memoryBlock in memoryMap:

                buff = ctxt.lookup(memoryBlock.name)
                assert isinstance(buff, VariableBuffer)
                if nodeMemoryConstraint is None:
                    size = buff.sizeInBytes()
                else:
                    mc = nodeMemoryConstraint.tensorMemoryConstraints[memoryBlock.name].memoryConstraints[memoryLevel]
                    if isinstance(buff, TransientBuffer):
                        size = mc.size
                    else:
                        size = mc.size * (buff._type.referencedType.typeWidth / 8) * mc.multiBufferCoefficient

                writer.writerow(
                    [memoryBlock.name,
                     str(memoryBlock.lifetime[0]),
                     str(memoryBlock.lifetime[1] + 1),
                     str(int(size))])

        try:
            minimallocInstallDir = os.environ["MINIMALLOC_INSTALL_DIR"]
        except KeyError:
            raise KeyError("MINIMALLOC_INSTALL_DIR symbol not found!")

        minimallocOutput = subprocess.run([
            f"{minimallocInstallDir}/minimalloc", f"--capacity={capacity}", f"--input={self._minimalloc_input}.csv",
            f"--output={self._minimalloc_output}.csv"
        ],
                                          capture_output = True,
                                          text = True)

        if minimallocOutput.returncode != 0:
            log.error(
                f"Memory allocator failed with return code {minimallocOutput.returncode} at memory level {memoryLevel} with capacity of {capacity} bytes!"
            )
            raise subprocess.CalledProcessError(minimallocOutput.returncode, " ".join(minimallocOutput.args))

        with open(f"{self._minimalloc_output}.csv", mode = "r", newline = "") as file:
            reader = csv.reader(file)
            header = next(reader)
            for row in reader:
                for memoryBlock in memoryMap:
                    if memoryBlock.name == row[0]:
                        memoryBlock._addrSpace = (int(row[-1]), int(row[-1]) + int(row[-2]))

        return memoryMap

    def computeTilingSchedule(self, ctxt: NetworkContext) -> TilingSolution:
        assert self.tilerModel is not None and self.symbolicMemoryConstraints is not None, "Set up the model before trying to compute a schedule!"
        collector = self.tilerModel.trySolveModel()
        tilingSolution = self._getTilingSolution(self.tilerModel, ctxt, collector, self.symbolicMemoryConstraints)
        if self.memoryAllocStrategy != "MiniMalloc":
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
            for memoryLevel in memoryMap.keys():
                constantTensorOffset = self.outerMemoryScheduler.getConstantTensorOffset(ctxt, memoryLevel)
                if memoryLevel == self.memoryHierarchy._defaultMemoryLevel.name:
                    memoryMap[memoryLevel][-1] = self.minimalloc(
                        memoryMap[memoryLevel][-1], ctxt, None,
                        self.memoryHierarchy.memoryLevels[memoryLevel].size - constantTensorOffset, memoryLevel)
                else:
                    for idx, memMap in enumerate(memoryMap[memoryLevel]):
                        if len(memoryMap[memoryLevel][idx]) != 0:
                            memoryMap[memoryLevel][idx] = self.minimalloc(
                                memMap, ctxt, tilingSolution[idx].nodeConstraints[0],
                                self.memoryHierarchy.memoryLevels[memoryLevel].size - constantTensorOffset, memoryLevel)
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
        # Transform schedule into type List[SubGraph] from Union[SubGraph, List[SubGraph]]
        schedule: List[SubGraph] = [[entry] if isinstance(entry, gs.Node) else entry for entry in schedule]
        tilerModel = TilerModel(searchStrategy = self.searchStrategy)
        tilerModel = self._setupGeometricConstraints(tilerModel, ctxt, schedule, layerBinding)
        tilerModel = self._setupTensorDimensionProducts(tilerModel, ctxt, schedule)
        tilerModel = self._setupHeuristics(tilerModel, ctxt, schedule)
        tilerModel, symbolicMemoryConstraints = self._setupMemoryConstraints(tilerModel, ctxt, schedule, layerBinding,
                                                                             targetMemoryLevelMapping)
        self.tilerModel = tilerModel
        self.symbolicMemoryConstraints = symbolicMemoryConstraints
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

    def _resolveTensorMemoryConstraint(self, tilerModel: TilerModel, ctxt: NetworkContext, collector: SolutionCollector,
                                       tensorMc: TensorMemoryConstraint) -> TensorMemoryConstraint:
        tensorName = tensorMc.tensorName
        resolvedTensorMc = TensorMemoryConstraint(tensorName, {}, ctxt)

        buffer = ctxt.lookup(tensorName)
        assert isinstance(buffer, VariableBuffer)

        for memoryLevel, mc in tensorMc.memoryConstraints.items():
            if isinstance(mc.size, int):
                resolvedMc = copy.copy(mc)
                resolvedMc.shape = tuple(buffer.shape)
            else:
                size = tilerModel._resolveVariable(mc.size)
                resolvedMc: MemoryConstraint = MemoryConstraint(memoryLevel, size)
                resolvedMc.multiBufferCoefficient = tilerModel._resolveVariable(mc.multiBufferCoefficient)

                if not isinstance(buffer, TransientBuffer):
                    _, copyIdx = tilerModel.getNameCopyIdx(mc.size.Name())
                    resolvedMc.shape = tuple(
                        tilerModel._resolveVariable(tilerModel.getTensorDimVar(tensorName, i, copyIdx))
                        for i in range(len(buffer.shape)))

            resolvedTensorMc.addMemoryConstraint(resolvedMc)

        return resolvedTensorMc

    def _getTilingSolution(self, tilerModel: TilerModel, ctxt: NetworkContext, collector: SolutionCollector,
                           allConstraints: List[PatternMemoryConstraints]) -> List[PatternMemoryConstraints]:

        def _checkResolve(ctxt, tensorName, tensorConstraint):
            return len(tensorConstraint.memoryConstraints) >= 2 or \
                    (not ctxt.is_global(tensorName) and isinstance(ctxt.lookup(tensorName), TransientBuffer))

        retList = []
        for patternConstraints in allConstraints:
            newMemoryConstraint = PatternMemoryConstraints()
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
            subGraphTensors = subGraph.tensors(check_duplicates = True)

            for tensor in subGraphTensors.values():
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
                opRepr = layerBinding[node.name].mapper.parser.operatorRepresentation
                tileConstraint: TileConstraint = layerBinding[node.name].mapper.binder.template.tileConstraint
                tilerModel = tileConstraint.addGeometricalConstraint(tilerModel, opRepr, ctxt)
                tilerModel = tileConstraint.addPolicyConstraint(tilerModel, opRepr, ctxt)
        return tilerModel

    def _setupHeuristics(self, tilerModel: TilerModel, ctxt: NetworkContext, schedule: List[SubGraph]) -> TilerModel:

        for idx, pattern in enumerate(schedule):
            subGraph = gs.Graph(nodes = pattern)
            subGraphTensors = subGraph.tensors(check_duplicates = True)

            patternMemSize = 0
            for name in subGraphTensors.keys():
                buffer = ctxt.lookup(name)
                assert isinstance(buffer, VariableBuffer)
                if not buffer._deploy:
                    continue
                patternMemSize += tilerModel.getTensorNumberOfEltVar(
                    name, copyIdx = idx) * buffer._type.referencedType.typeWidth // 8

            _max = patternMemSize if isinstance(patternMemSize, int) else patternMemSize.Max()

            patternVariable = tilerModel.addVariable(name = "DEEPLOY_PATTERN_MEM",
                                                     lowerBound = 1,
                                                     upperBound = _max,
                                                     copyIdx = idx)
            tilerModel.addConstraint(patternVariable == patternMemSize)
            tilerModel.addObjective(Objective(var = patternVariable, optDir = Objective.OptimizationDirection.Maximize))

        return tilerModel

    def _setupMemoryConstraints(
            self, tilerModel: TilerModel, ctxt: NetworkContext, schedule: List[SubGraph],
            layerBinding: OrderedDict[str, ONNXLayer],
            targetMemoryLevelMapping: TargetMemoryLevelMapping) -> Tuple[TilerModel, List[PatternMemoryConstraints]]:

        allMemoryConstraints = self._generateMemoryConstraints(tilerModel, ctxt, schedule, layerBinding,
                                                               targetMemoryLevelMapping)

        outerMemoryConstraints = PatternMemoryConstraints()
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

    def _generateMemoryConstraints(
            self, tilerModel: TilerModel, ctxt: NetworkContext, schedule: List[SubGraph],
            layerBinding: OrderedDict[str, ONNXLayer],
            targetMemoryLevelMapping: TargetMemoryLevelMapping) -> List[PatternMemoryConstraints]:
        # SCHEREMO: Construct non-double-buffered constraints of local variable buffers
        globalConstantBufferConstraints = self._generateConstantBufferConstraints(ctxt)

        outerAnalysis = OuterBufferLivenessAnalysis(ctxt)
        outerLive = outerAnalysis.initLive(layerBinding)

        patternMcs: List[PatternMemoryConstraints] = []
        for idx, pattern in enumerate(schedule):
            outerKill = outerAnalysis.computeKill(pattern)
            outerGen = outerAnalysis.computeGen(pattern)

            outerNodeMc = NodeMemoryConstraint()
            for tensor in outerLive:
                outerNodeMc.addTensorConstraint(self._constraintFromBuffer(tensor, ctxt), "input")
            for tensor in outerGen:
                outerNodeMc.addTensorConstraint(self._constraintFromBuffer(tensor, ctxt), "output")

            outerLive = outerAnalysis.computeLive(outerLive, outerGen, outerKill)

            outerNodeMc += globalConstantBufferConstraints

            tilerModel.copyIdx = idx
            innerAnalysis = InnerBufferLivenessAnalysis(ctxt, pattern)
            innerLive = innerAnalysis.initLive()
            for node in pattern:
                innerKill = innerAnalysis.computeKill(node)
                innerGen = innerAnalysis.computeGen(node)

                innerNodeMc = NodeMemoryConstraint()
                for tensor in innerLive:
                    memory = targetMemoryLevelMapping.lookup(node.name, tensor)
                    tensorMc = self._constraintFromTiling(tensor, tilerModel, idx, memory, ctxt)
                    innerNodeMc.addTensorConstraint(tensorMc, "input")
                for tensor in innerGen:
                    memory = targetMemoryLevelMapping.lookup(node.name, tensor)
                    tensorMc = self._constraintFromTiling(tensor, tilerModel, idx, memory, ctxt)
                    innerNodeMc.addTensorConstraint(tensorMc, "output")

                innerLive = innerAnalysis.computeLive(innerLive, innerGen, innerKill)

                # Addition creates a new NodeMemoryConstraint object
                nodeMc = outerNodeMc + innerNodeMc

                self._generateIntermediateTilingSteps(tilerModel, ctxt, nodeMc, pattern)

                nodeMc += self._generatePatternStepTransientBufferConstraints(tilerModel, ctxt, layerBinding, node,
                                                                              targetMemoryLevelMapping)

            patternMc = PatternMemoryConstraints()
            patternMc.addConstraint(nodeMc)
            patternMcs.append(patternMc)

        return patternMcs

    def _generateIntermediateTilingSteps(self, tilerModel: TilerModel, ctxt: NetworkContext,
                                         nodeMc: NodeMemoryConstraint, pattern: SubGraph) -> None:
        for tensor, tensorMc in nodeMc.tensorMemoryConstraints.items():
            assert len(tensorMc.memoryConstraints) >= 1 and len(tensorMc.memoryConstraints) <= 2
            if len(tensorMc.memoryConstraints) == 1:
                continue

            srcMc, dstMc = list(tensorMc.memoryConstraints.values())
            assert isinstance(srcMc.size, int), "Source and destination are not ordered properly."
            path = self.memoryHierarchy.pathSearch(srcMc.memoryLevel, dstMc.memoryLevel)
            assert len(path) >= 2, f"No path found between {srcMc.memoryLevel} and {dstMc.memoryLevel}"
            mcs: OrderedDict[str, MemoryConstraint] = OrderedDict()
            mcs[srcMc.memoryLevel] = srcMc
            for memory in path[1:]:
                coeff = self.multiBufferStrategy(tilerModel, ctxt, pattern, path, memory, tensor)
                assert isinstance(coeff, int) and coeff > 0, \
                        f"MultiBuffer coefficient should be an integer higher then 0. Received invalid coefficient={coeff}."
                mc = MemoryConstraint(memory, dstMc.size)
                mc.multiBufferCoefficient = coeff
                mcs[memory] = mc
            tensorMc.memoryConstraints = mcs

    def _generateConstantBufferConstraints(self, ctxt: NetworkContext) -> NodeMemoryConstraint:
        nodeMc: NodeMemoryConstraint = NodeMemoryConstraint()
        globalConstantBuffers: List[ConstantBuffer] = [
            obj for obj in ctxt.globalObjects.values() if isinstance(obj, ConstantBuffer) and obj._deploy
        ]

        for buffer in globalConstantBuffers:
            memoryConstraint = MemoryConstraint(buffer._memoryLevel, buffer.sizeInBytes())
            tensorConstraint = TensorMemoryConstraint(buffer.name, {buffer._memoryLevel: memoryConstraint}, ctxt)
            nodeMc.addTensorConstraint(tensorConstraint, "input")
        return nodeMc

    def _constraintFromBuffer(self, name: str, ctxt: NetworkContext) -> TensorMemoryConstraint:
        buffer = ctxt.lookup(name)
        assert isinstance(buffer, VariableBuffer)
        memory = buffer._memoryLevel
        size = math.prod(buffer.shape)
        return TensorMemoryConstraint(name, {memory: MemoryConstraint(memory, size)}, ctxt)

    def _constraintFromTiling(self, name: str, tilerModel: TilerModel, copyIdx: int, memory: str,
                              ctxt: NetworkContext) -> TensorMemoryConstraint:
        if tilerModel.checkTensorExists(name, copyIdx):
            size = tilerModel.getTensorNumberOfEltVar(name, copyIdx)
            return TensorMemoryConstraint(name, {memory: MemoryConstraint(memory, size)}, ctxt)
        else:
            return self._constraintFromBuffer(name, ctxt)

    def _generatePatternStepTransientBufferConstraints(
            self, tilerModel: TilerModel, ctxt: NetworkContext, layerBinding: OrderedDict[str, ONNXLayer],
            step: gs.Node, targetMemoryLevelMapping: TargetMemoryLevelMapping) -> NodeMemoryConstraint:
        patternStepTransientBufferSizes = NodeMemoryConstraint()

        template = layerBinding[step.name].mapper.binder.template
        opRepr = layerBinding[step.name].mapper.parser.operatorRepresentation

        symbolicOpRepr = template.tileConstraint.constructSymbolicNodeRep(tilerModel, parseDict = opRepr, ctxt = ctxt)
        for name, size in template.computeTransientBuffersSize(ctxt, symbolicOpRepr):
            # SCHEREMO: Assume transientbuffers end up in the same level as their user's main input
            memoryLevel = targetMemoryLevelMapping.lookup(step.name, step.inputs[0].name)
            ctxt.lookup(name)._memoryLevel = memoryLevel
            varSize = tilerModel.addTransientBufferSizeToModel(name, size)
            memoryConstraint = MemoryConstraint(memoryLevel, varSize)
            tensorConstraint = TensorMemoryConstraint(name, {memoryLevel: memoryConstraint}, ctxt)
            patternStepTransientBufferSizes.addTensorConstraint(tensorConstraint, "intermediate")

        return patternStepTransientBufferSizes

    def assertLayerWiseTiling(self, schedule: List[List[gs.Node]]) -> bool:
        return all(len(pattern) == 1 for pattern in schedule)

    def assertUniformMemoryLevelAllocation(self, ctxt: NetworkContext, memoryLevel: str) -> bool:
        return all(buffer._memoryLevel == memoryLevel for buffer in ctxt.localObjects.values())

    def testTilingSolutionCorrectness(self, tilingSolution: TilingSolution) -> None:
        # LMACAN: Assert buffer sizes are word aligned as per comment in MemoryScheduler.py:MemoryScheduler._buildCostVector()
        byteAlignment = MemoryScheduler.byteAlignment
        for patternMemoryConstraint in tilingSolution:
            for nodeMemoryConstraint in patternMemoryConstraint.nodeConstraints:
                for tensorMemoryConstraint in nodeMemoryConstraint.tensorMemoryConstraints.values():
                    for memoryConstraint in tensorMemoryConstraint.memoryConstraints.values():
                        if memoryConstraint.addrSpace is not None:
                            assert isinstance(memoryConstraint.multiBufferCoefficient, int)
                            bufferSize = (memoryConstraint.addrSpace[1] -
                                          memoryConstraint.addrSpace[0]) // memoryConstraint.multiBufferCoefficient
                            assert bufferSize % byteAlignment == 0, f"Buffer in {memoryConstraint} is not {byteAlignment} byte aligned"

    def testMemoryMapCorrectness(self, memoryMap: Dict[str, List[List[MemoryBlock]]], graph: gs.Graph,
                                 schedule: Schedule) -> None:

        memoryBlockMap = {
            memoryBlock.name: memoryBlock for levelMemoryMap in memoryMap.values() for memoryBlock in levelMemoryMap[-1]
        }

        # JUNGVI: Assert output buffers are alive until the end
        for tensor in graph.outputs:
            assert memoryBlockMap[tensor.name]._lifetime[-1] == len(
                schedule), "Invalid memory map! Output buffer is not alive at the last step!"

        # JUNGVI: Assert input buffers are alive at the beginning
        for inputBuffer in graph.inputs:
            assert memoryBlockMap[
                inputBuffer.name]._lifetime[0] == 0, "Invalid memory map! Input buffer is not alive at step 0!"

        # JUNGVI: Assert that at every computation step, the required buffers are alive somewhere in memory
        for stepIdx, pattern in enumerate(schedule):
            node = pattern[0]
            nodeIO = [node for node in node.inputs + node.outputs if not isinstance(node, gs.Constant)]
            for tensor in nodeIO:
                lifetime = memoryBlockMap[tensor.name]._lifetime
                assert stepIdx in range(lifetime[0], lifetime[-1] +
                                        1), f"Invalid memory map! Buffer {tensor.name} is not alive at step {stepIdx}!"


class TilerDeployerWrapper(NetworkDeployerWrapper):

    def __init__(self,
                 deployer: Union[MemoryLevelAwareDeployer, MemoryDeployerWrapper],
                 tilerCls: Type[Tiler] = Tiler,
                 testName: Optional[str] = None,
                 workDir: Optional[str] = None):
        super().__init__(deployer)
        assert isinstance(self.Platform, (MemoryPlatform, MemoryPlatformWrapper)), \
            f"Platform should be a MemoryPlatform or MemoryPlatformWrapper! Got {type(self.Platform).__name__}"
        self.tiler = tilerCls(self.Platform.memoryHierarchy, testName = testName, workDir = workDir)

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
