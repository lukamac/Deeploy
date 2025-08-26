# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import onnx_graphsurgeon as gs

from .graphDiffUtils import DiffTree, createParentDiffNode, listDiff, nodeDiff, tensorDiff


def removeBatching(
    test_inputs_files, test_outputs_files, activations_files, graph: gs.Graph
) -> Tuple[Dict[str, npt.NDArray], Dict[str, npt.NDArray], Optional[Dict[str, npt.NDArray]], gs.Graph]:
    tensors = graph.tensors()

    for tensor in tensors.values():
        if isinstance(tensor, gs.Constant):
            continue
        if tensor.shape[0] == 10:
            tensor.shape = [1] + tensor.shape[1:]

    test_inputs_files = {key: value[:1] for key, value in test_inputs_files.items()}
    test_outputs_files = {key: value[:1] for key, value in test_outputs_files.items()}
    if activations_files is not None:
        activations_files = {key: value[:1] for key, value in activations_files.items()}

    return test_inputs_files, test_outputs_files, activations_files, graph


def generateDebugConfig(test_inputs_files, test_outputs_files, activations_files,
                        graph: gs.Graph) -> Tuple[List[npt.NDArray], List[npt.NDArray], gs.Graph]:
    test_inputs_files, test_outputs_files, activations_files, graph = removeBatching(
        test_inputs_files, test_outputs_files, activations_files, graph)

    # Choose nodes
    graph.nodes = graph.nodes[:2]
    graph.outputs = list(graph.nodes[-1].outputs)
    graph.cleanup(remove_unused_graph_inputs = True, remove_unused_node_outputs = True)

    # Fixup inputs
    #graph.inputs = list(tensor for tensor in graph.nodes[0].inputs if not isinstance(tensor, gs.Constant))

    test_input_values = []
    for tensor in graph.inputs:
        if tensor.name in test_inputs_files:
            test_input_values.append(test_inputs_files[tensor.name])
        elif activations_files is not None and tensor.name in activations_files:
            test_input_values.append(activations_files[tensor.name])
        else:
            test_input_values.append(np.random.rand(*tensor.shape))

    test_output_values = []
    for tensor in graph.outputs:
        if tensor.name in test_outputs_files:
            test_output_values.append(test_outputs_files[tensor.name])
        elif activations_files is not None and tensor.name in activations_files:
            test_output_values.append(activations_files[tensor.name])
        else:
            test_output_values.append(np.random.rand(*tensor.shape))

    graph.cleanup(remove_unused_graph_inputs = True, remove_unused_node_outputs = True)
    graph.toposort()

    test_inputs = [x.reshape(-1).astype(np.float64) for x in test_input_values]
    test_outputs = [x.reshape(-1).astype(np.float64) for x in test_output_values]

    return test_inputs, test_outputs, graph


def graphDiff(graph: gs.Graph, other: gs.Graph) -> DiffTree:
    graph = graph.toposort()
    other = other.toposort()
    diffs = []
    diffs.append(listDiff(graph.nodes, other.nodes, "nodes", nodeDiff))
    diffs.append(listDiff(graph.inputs, other.inputs, "inputs", tensorDiff))
    diffs.append(listDiff(graph.outputs, other.outputs, "outputs", tensorDiff))
    root = createParentDiffNode(graph, other, graph.name, diffs)
    return DiffTree(root)
