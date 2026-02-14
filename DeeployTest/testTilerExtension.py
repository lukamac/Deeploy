# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Dict, List

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest
from testUtils.platformMapping import mapDeployer, mapPlatform, setupMemoryPlatform
from testUtils.testRunner import TestGeneratorArgumentParser
from testUtils.typeMapping import inferTypeAndOffset

from Deeploy.DeeployTypes import NetworkContext, Schedule, TransientBuffer, VariableBuffer
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.MemoryLevelExtension.NetworkDeployers.MemoryLevelDeployer import MemoryDeployerWrapper
from Deeploy.MemoryLevelExtension.OptimizationPasses.MemoryLevelAnnotationPasses import AnnotateDefaultMemoryLevel, \
    AnnotateIOMemoryLevel
from Deeploy.TilingExtension.MemoryConstraints import TensorMemoryConstraint
from Deeploy.TilingExtension.TilerExtension import TilerDeployerWrapper, TilingSolution


def _mockScheduler(graph: gs.Graph) -> List[List[gs.Node]]:
    """Mock of the Global Scheduler's inteface

    Returns a list of list of nodes instead of simply a list.
    Inner list represent the patter over which we tile.
    """
    return [[node] for node in graph.nodes]


# TODO: Remove this function in favour of the ones that are implemented in Deeploy
def getMemoryOccupation(ctxt: NetworkContext, tensorMemoryConstraints: Dict[str, TensorMemoryConstraint],
                        memoryLevel: str):
    occupation = 0

    for buffer in ctxt.globalObjects.values():
        if not isinstance(buffer, VariableBuffer):
            continue
        if buffer._memoryLevel == memoryLevel and len(buffer._users) > 0 and buffer._deploy:
            occupation += buffer.sizeInBytes()

    for name, tensorMemoryConstraint in tensorMemoryConstraints.items():
        if memoryLevel not in tensorMemoryConstraint.memoryConstraints:
            continue

        mc = tensorMemoryConstraint.memoryConstraints[memoryLevel]

        buffer = ctxt.lookup(name)
        if isinstance(buffer, TransientBuffer):
            typeWidth = 1
        else:
            typeWidth = (buffer._type.referencedType.typeWidth // 8)

        occupation += mc.multiBufferCoefficient * mc.size * typeWidth

    return occupation


def validateSolution(schedule: Schedule, tilingSolution: TilingSolution, memoryHierarchy: MemoryHierarchy,
                     ctxt: NetworkContext):

    assert len(schedule) == len(tilingSolution), "ERROR: schedule and tilingSchedule don't have the same length"

    for pattern, patternMemoryConstraint in zip(schedule, tilingSolution):

        # Collect all deployed tensors
        patternTensors = set()
        for node in pattern:
            for tensor in node.inputs + node.outputs:
                if ctxt.lookup(tensor.name)._deploy:
                    patternTensors.add(tensor.name)

        nodeNames = {node.name for node in pattern}
        usedTensors = {t for t in patternTensors if not nodeNames.isdisjoint(ctxt.lookup(t)._users)}
        producedTensors = {tensor.name for node in pattern for tensor in node.outputs}

        for nodeMemoryConstraint in patternMemoryConstraint.nodeConstraints:
            borderTensors = {
                tensor.tensorName
                for tensor in nodeMemoryConstraint.tensorMemoryConstraints.values()
                if len(tensor.memoryConstraints) > 1
            }

            intermediateTensors = patternTensors - borderTensors

            assert intermediateTensors == usedTensors & producedTensors, \
                    "ERROR in tilingSchedule!"
            assert borderTensors == usedTensors ^ producedTensors, \
                    "ERROR in tilingSchedule!"

            l1Occupation = getMemoryOccupation(ctxt, nodeMemoryConstraint.tensorMemoryConstraints, "L1")
            assert l1Occupation <= memoryHierarchy.memoryLevels['L1'].size, "L1 usage is too high!"

            l2Occupation = getMemoryOccupation(ctxt, nodeMemoryConstraint.tensorMemoryConstraints, "L2")
            assert l2Occupation <= memoryHierarchy.memoryLevels['L2'].size, "L2 usage is too high!"


def setupDeployer(memoryHierarchy: MemoryHierarchy, graph: gs.Graph) -> TilerDeployerWrapper:
    inputTypes = {}
    inputOffsets = {}

    _DEEPLOYSTATEDIR = os.path.join(args.dumpdir, "deeployStates")

    inputs = np.load(f'./{args.dir}/inputs.npz')
    tensors = graph.tensors()

    # Load as float64 and infer types later
    test_inputs = [inputs[x].reshape(-1).astype(np.float64) for x in inputs.files]

    platform, signProp = mapPlatform(args.platform)

    for index, num in enumerate(test_inputs):
        _type, offset = inferTypeAndOffset(num, signProp)
        inputTypes[f"input_{index}"] = _type
        inputOffsets[f"input_{index}"] = offset
        if "simpleRegression" in args.dir:
            inputOffsets[f"input_{index}"] = 0

    deployer = mapDeployer(platform,
                           graph,
                           inputTypes,
                           deeployStateDir = _DEEPLOYSTATEDIR,
                           inputOffsets = inputOffsets,
                           scheduler = _mockScheduler)

    memoryLevelAnnotationPasses = [AnnotateIOMemoryLevel("L2"), AnnotateDefaultMemoryLevel(memoryHierarchy)]

    # Make the platform memory-level aware
    deployer.Platform = setupMemoryPlatform(deployer.Platform,
                                            memoryHierarchy,
                                            defaultTargetMemoryLevel = memoryHierarchy.memoryLevels["L1"])
    # Make the deployer memory-level aware
    deployer = MemoryDeployerWrapper(deployer, memoryLevelAnnotationPasses)

    # Make the deployer tiler aware
    deployer = TilerDeployerWrapper(deployer)

    deployer.frontEnd()

    return deployer


if __name__ == '__main__':
    parser = TestGeneratorArgumentParser(description = "Test Utility for the Tiler Extension.")
    parser.add_argument('--l1', type = int, default = 64000, help = 'Set L1 size\n')
    parser.add_argument('--l2', type = int, default = 512000, help = 'Set L2 size\n')
    parser.add_argument('--shouldFail', action = 'store_true', default = False)
    args = parser.parse_args()

    onnx_graph = onnx.load_model(f'./{args.dir}/network.onnx')
    graph = gs.import_onnx(onnx_graph)

    # Instantiate Classes Requried for Memory Level Annotation Extension
    L3_2 = MemoryLevel(name = "L3.1", neighbourNames = ["L2"], size = 1024000)
    L3_1 = MemoryLevel(name = "L3.2", neighbourNames = ["L2"], size = 4000)
    L2 = MemoryLevel(name = "L2", neighbourNames = ["L3.1", "L3.2", "L1"], size = args.l2)
    L1 = MemoryLevel(name = "L1", neighbourNames = ["L2"], size = args.l1)

    memoryHierarchy = MemoryHierarchy([L3_1, L3_2, L2, L1])
    memoryHierarchy.setDefaultMemoryLevel("L2")

    deployer = setupDeployer(memoryHierarchy, graph)

    if args.shouldFail:
        with pytest.raises(Exception):
            tilingSolution = deployer.tiler.computeTilingSchedule(deployer.ctxt)

        print("Tiler test ended, failed as expected!")
    else:
        _ = deployer.generateFunction()

        tiler = deployer.tiler
        tilerModel = tiler.tilerModel
        symbolicMemoryConstraints = tiler.symbolicMemoryConstraints

        assert tilerModel is not None, "The tiler model is undefined"
        assert tilerModel._collector is not None, "The constraint problem hasn't been solved"
        assert symbolicMemoryConstraints is not None, "The tiler's symbolic memory constraints are undefined"
        tilingSolution = tiler._getTilingSolution(tilerModel, deployer.ctxt, tilerModel._collector,
                                                  symbolicMemoryConstraints)

        schedule = _mockScheduler(deployer.graph)
        validateSolution(schedule, tilingSolution, memoryHierarchy, deployer.ctxt)
        print("Tiler test ended, no memory violations!")
