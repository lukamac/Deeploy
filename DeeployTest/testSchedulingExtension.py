# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest
from testUtils.platformMapping import mapDeployer, mapPlatform, setupMemoryPlatform
from testUtils.testRunner import TestGeneratorArgumentParser
from testUtils.typeMapping import inferTypeAndOffset

from Deeploy.DeeployTypes import NetworkContext, NetworkDeployer, ONNXLayer, Schedule, TransientBuffer, VariableBuffer
from Deeploy.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.NetworkDeployers.MemoryLevelDeployer import MemoryDeployerWrapper
from Deeploy.OptimizationPasses.MemoryLevelAnnotationPasses import AnnotateDefaultMemoryLevel, AnnotateIOMemoryLevel
from Deeploy.TilingExtension.MemoryScheduler import MemoryBlock
from Deeploy.TilingExtension.TilerExtension import TilerDeployerWrapper, TilingSolution


# Mock of the Global Scheduler's inteface
# Returns a list of list of nodes instead of simply a list
# Inner list represent the patter over which we tile
def _mockScheduler(graph: gs.Graph) -> List[List[gs.Node]]:

    schedule = [[node] for node in graph.nodes]

    return schedule


def _filterSchedule(schedule: List[List[gs.Node]], layerBinding: 'OrderedDict[str, ONNXLayer]') -> List[List[gs.Node]]:

    filteredSchedule = []

    for pattern in schedule:

        filteredSchedulePattern = []
        for node in pattern:
            if node.name in layerBinding.keys():
                filteredSchedulePattern.append(node)
        filteredSchedule.append(filteredSchedulePattern)

    return filteredSchedule


def getMemoryOccupation(ctxt, tiledTensors, memoryLevel):

    occupation = 0

    for tensor in tiledTensors.values():
        for memoryConstraint in tensor.memoryConstraints.values():
            if memoryConstraint.memoryLevel == memoryLevel:

                if not isinstance(ctxt.lookup(tensor.tensorName), TransientBuffer):
                    typeWidth = (ctxt.lookup(tensor.tensorName)._type.referencedType.typeWidth // 8)
                else:
                    typeWidth = 1

                delta = memoryConstraint.multiBufferCoefficient * memoryConstraint.size * typeWidth
                occupation += delta

    return occupation


def validateTilingTopologySolution(schedule: Schedule, tilingSchedule: Schedule, memoryHierarchy: MemoryHierarchy):

    assert len(schedule) == len(tilingSchedule), "ERROR: schedule and tilingSchedule don't have the same length"

    for pattern, tilingPattern in zip(schedule, tilingSchedule):
        subGraph = gs.Graph(nodes = pattern)
        patternTensors = set([key for key, value in subGraph.tensors().items() if ctxt.lookup(key)._deploy])

        # intermediateTensors are all tensors that are used and produced by the pattern.
        # Including transient Buffers!
        usedTensors = set()
        producedTensors = set()
        transientTensors = set()

        for tensor in patternTensors:
            users = ctxt.lookup(tensor)._users

            for node in pattern:
                if node.name in users:
                    usedTensors.add(tensor)
                    break

        for node in pattern:
            outputTensors = {node.name for node in node.outputs}
            producedTensors |= outputTensors

        for tensorName, varBuffer in ctxt.localObjects.items():
            if isinstance(varBuffer, TransientBuffer):
                assert len(varBuffer._users) == 1
                if varBuffer._users[0] in patternTensors:
                    transientTensors.add(tensorName)

        for tilingStep in tilingPattern.nodeConstraints:
            borderTensors = {
                tensor.tensorName
                for tensor in tilingStep.tensorMemoryConstraints.values()
                if len(tensor.memoryConstraints.keys()) > 1
            }

            intermediateTensors = patternTensors - borderTensors
            assert intermediateTensors == ((usedTensors & producedTensors) |
                                           transientTensors), "ERROR in tilingSchedule!"

            assert borderTensors == (usedTensors - producedTensors) | (producedTensors -
                                                                       usedTensors), "ERROR in tilingSchedule!"

            l1Occupation = getMemoryOccupation(ctxt, tilingStep.tensorMemoryConstraints, "L1")
            assert l1Occupation <= memoryHierarchy.memoryLevels['L1'].size, "L1 usage is too high"

            l2Occupation = getMemoryOccupation(ctxt, tilingStep.tensorMemoryConstraints, "L2")
            assert l2Occupation <= memoryHierarchy.memoryLevels['L2'].size, "L2 usage is too high!"


def _findBlocks(memoryMap: Dict[str, List[List[MemoryBlock]]], name: str) -> List[MemoryBlock]:
    return [block for patterns in memoryMap.values() for pattern in patterns for block in pattern if block.name == name]


def validateStaticMemoryLayoutSolution(ctxt: NetworkContext, memoryMap: Dict[str, List[List[MemoryBlock]]]):
    # SCHEREMO: Assert that every VariableBuffer and ConstantBuffer is fully allocated somewhere
    # SCHEREMO: This doesn't need to hold for depth-first tiling!
    for name, buff in ctxt.localObjects.items():
        if not isinstance(buff, VariableBuffer):
            continue

        # SCHEREMO: Exception for memory arenas
        if len(buff._users) == 0:
            continue

        blocks = _findBlocks(memoryMap, name)
        assert len(blocks) > 0, f"Found no blocks for buffer {name}"

        mismatched = [b for b in blocks if b.addrSpace is None or b.addrSpace.size != buff.sizeInBytes()]
        if len(mismatched) > 0:
            for block in mismatched:
                print(f"Buffer {name} of size {buff.sizeInBytes()} has mismatching block allocations:")
                if block.addrSpace is None:
                    print(f"  - {block.name}: address space was not allocated")
                elif block.addrSpace.size > buff.sizeInBytes():
                    print(f"  - {block.name}: too much space allocated")
                elif block.addrSpace.size < buff.sizeInBytes():
                    print(f"  - {block.name}: not enough space allocated")
            raise RuntimeError(f"Buffer {name} has mismatching block allocations")


def validateDynamicMemoryLayoutSolution(ctxt: NetworkContext, tilingSchedule: TilingSolution,
                                        memoryMap: Dict[str, List[List[MemoryBlock]]]):

    # SCHEREMO: Assert that tilingSchedule is implemented
    for patternIdx, patternConstraints in enumerate(tilingSchedule):
        for nodeConstraint in patternConstraints.nodeConstraints:
            for tensorConstraint in nodeConstraint.tensorMemoryConstraints.values():

                blocks = _findBlocks(memoryMap, tensorConstraint.tensorName)
                blockLevels = [block.level for block in blocks]

                buff = ctxt.lookup(tensorConstraint.tensorName)

                for memoryConstraint in tensorConstraint.memoryConstraints.values():
                    # SCHEREMO: Don't check static allocation
                    if buff._memoryLevel == memoryConstraint.memoryLevel:
                        continue

                    assert memoryConstraint.memoryLevel in blockLevels, f"No constraint for {tensorConstraint.tensorName} memoryLevel {memoryConstraint.memoryLevel}"

                    patternBlocks = memoryMap[memoryConstraint.memoryLevel][patternIdx]

                    tensorBlocks = [block for block in patternBlocks if block.name == tensorConstraint.tensorName]
                    assert len(
                        tensorBlocks) == 1, f"{tensorConstraint.tensorName} not exactly once in pattern {patternIdx}!"

                    block = tensorBlocks[0]

                    collisions = [
                        other.name
                        for other in patternBlocks
                        if other != block and not ctxt.isAliased(block.name, other.name) and block.collides(other)
                    ]
                    assert len(
                        collisions
                    ) > 0, f"Block {block.name} has collisions in pattern {patternIdx}. Collisions: {collisions}"

                    ctxtSize = memoryConstraint.size * memoryConstraint.multiBufferCoefficient * (
                        buff._type.referencedType.typeWidth // 8)
                    assert block.addrSpace is not None, f"Expected block {block.name} to have an assigned address space."
                    assert ctxtSize <= block.addrSpace.size, (
                        f"{tensorConstraint.tensorName}'s expected size does not fit into the block's allocated address space!"
                        f"Expected size {ctxtSize} > allocated address space {block.addrSpace.size}")


def setupDeployer(memoryHierarchy: MemoryHierarchy, graph: gs.Graph) -> NetworkDeployer:

    inputTypes = {}
    inputOffsets = {}

    _DEEPLOYSTATEDIR = os.path.join(args.dumpdir, "deeployStates")

    inputs = np.load(f'./{args.dir}/inputs.npz')
    tensors = graph.tensors()

    # Load as int64 and infer types later
    test_inputs = [inputs[x].reshape(-1).astype(np.int64) for x in inputs.files]

    platform, signProp = mapPlatform(args.platform)

    for index, num in enumerate(test_inputs):
        _type, offset = inferTypeAndOffset(num, signProp)
        inputTypes[f"input_{index}"] = _type
        inputOffsets[f"input_{index}"] = offset

    deployer = mapDeployer(platform,
                           graph,
                           inputTypes,
                           deeployStateDir = _DEEPLOYSTATEDIR,
                           inputOffsets = inputOffsets,
                           scheduler = _mockScheduler)

    memoryLevelAnnotationPasses = [AnnotateDefaultMemoryLevel(memoryHierarchy), AnnotateIOMemoryLevel("L2")]

    # Make the deployer memory-level aware
    deployer.Platform = setupMemoryPlatform(deployer.Platform,
                                            memoryHierarchy,
                                            defaultTargetMemoryLevel = memoryHierarchy.memoryLevels["L1"])
    # Make the deployer memory-level aware
    deployer = MemoryDeployerWrapper(deployer, memoryLevelAnnotationPasses)

    # Make the deployer tiler aware
    deployer = TilerDeployerWrapper(deployer)

    deployer.frontEnd()
    #deployer.midEnd()

    return deployer


def validateEffectiveLoad(outerMemoryMap: Dict[str, List[List[MemoryBlock]]],
                          innerMemoryMap: Dict[str, List[List[MemoryBlock]]], memoryHierarchy: MemoryHierarchy):

    def perMemLoad(memMap: Dict[str, List[List[MemoryBlock]]]) -> Dict[str, int]:
        load = {}
        for memory, patterns in memMap.items():
            maxAddr = 0
            for pattern in patterns:
                for block in pattern:
                    assert block.addrSpace is not None, f"Expected block {block.name} to have an allocated address space."
                    maxAddr = max(maxAddr, block.addrSpace.end)
            load[memory] = maxAddr
        return load

    staticMemoryLoad = perMemLoad(outerMemoryMap)
    dynamicMemoryLoad = perMemLoad(innerMemoryMap)

    totalMemoryLoad = {}
    for level in dynamicMemoryLoad.keys():
        totalMemoryLoad[level] = staticMemoryLoad[level] + dynamicMemoryLoad[level]

    for level, load in totalMemoryLoad.items():
        assert memoryHierarchy.memoryLevels[level].size > load, \
            f"Effective memory layout does not fit {memoryHierarchy.memoryLevels[level].size} in {level}"


def validateDynamicLifetimes(ctxt: NetworkContext, tilingSchedule: TilingSolution,
                             outerMemoryMap: Dict[str, List[List[MemoryBlock]]]):

    for patternIdx, pattern in enumerate(tilingSchedule):
        for nodeConstraint in pattern.nodeConstraints:
            for tensor in nodeConstraint.tensorMemoryConstraints.values():
                name = tensor.tensorName

                buf = ctxt.lookup(name)
                if isinstance(buf, TransientBuffer) or ctxt.is_global(name):
                    continue

                blocks = _findBlocks(outerMemoryMap, name)
                assert len(blocks) == 1, f"Found {name} more than once in static life time map!"

                block = blocks[0]
                assert block.lifetime.contains(patternIdx), \
                        f"Tile of {name} is used after deallocation of the static buffer!"


if __name__ == '__main__':

    parser = TestGeneratorArgumentParser(description = "Test Utility for the Scheduling Extension.")
    parser.add_argument('--l1', metavar = 'l1', dest = 'l1', type = int, default = 64000, help = 'Set L1 size\n')
    parser.add_argument('--shouldFail', action = 'store_true')
    parser.set_defaults(shouldFail = False)
    args = parser.parse_args()

    onnx_graph = onnx.load_model(f'./{args.dir}/network.onnx')
    graph = gs.import_onnx(onnx_graph)

    # Instantiate Classes Requried for Memory Level Annotation Extension
    L3_2 = MemoryLevel(name = "L3.1", neighbourNames = ["L2"], size = 1024000)
    L3_1 = MemoryLevel(name = "L3.2", neighbourNames = ["L2"], size = 4000)
    L2 = MemoryLevel(name = "L2", neighbourNames = ["L3.1", "L3.2", "L1"], size = 512000)
    L1 = MemoryLevel(name = "L1", neighbourNames = ["L2"], size = args.l1)

    memoryHierarchy = MemoryHierarchy([L3_1, L3_2, L2, L1])
    #memoryHierarchy.setDefaultMemoryLevel("L3.1")
    memoryHierarchy.setDefaultMemoryLevel("L2")

    deployer = setupDeployer(memoryHierarchy, graph)

    schedule = _filterSchedule(_mockScheduler(graph), deployer.layerBinding)

    if args.shouldFail:
        with pytest.raises(Exception):
            tilingSchedule = deployer.tiler.computeTilingSchedule(deployer.ctxt)

        print("Tiler test ended, failed as expected!")
    else:

        _ = deployer.generateFunction()

        tilingSchedule = deployer.tiler._getTilingSolution(deployer.tiler.tilerModel, deployer.ctxt,
                                                           deployer.tiler.tilerModel._collector,
                                                           deployer.tiler.symbolicMemoryConstraints)

        ctxt = deployer.ctxt
        layerBinding = deployer.layerBinding
        schedule = _mockScheduler(deployer.graph)

        validateTilingTopologySolution(schedule, tilingSchedule, memoryHierarchy)

        innerMemoryMap = deployer.tiler.innerMemoryScheduler.memoryMap
        outerMemoryMap = deployer.tiler.outerMemoryScheduler.memoryMap

        validateStaticMemoryLayoutSolution(ctxt, outerMemoryMap)
        validateDynamicMemoryLayoutSolution(ctxt, tilingSchedule, innerMemoryMap)

        validateDynamicLifetimes(ctxt, tilingSchedule, outerMemoryMap)

        validateEffectiveLoad(outerMemoryMap, innerMemoryMap, memoryHierarchy)

        print("Tiler test ended, no memory violations!")
