import numpy as np
import onnx
import onnx_graphsurgeon as gs

batch = 1
channel_in = 8
channel_out = 16
height = 100
width = 10

size_in = batch * channel_in * height * width
size_out = batch * channel_out * height * width
size_weights = channel_out * channel_in

input = np.asarray([1] * size_in).reshape(batch, channel_in, height, width).astype(np.float64)
weights = np.asarray([1] * size_weights).reshape(channel_out, channel_in, 1, 1).astype(np.float64)
output = np.asarray([channel_in] * size_out).reshape(batch, channel_out, height, width).astype(np.float64)

np.savez("inputs.npz", input = input)
np.savez("outputs.npz", output = output)

input_tensor = gs.Variable(
    name = "input",
    dtype = np.float64,
    shape = input.shape,
)

weights_tensor = gs.Constant(
    name = "weights",
    values = weights,
)

conv_output_tensor = gs.Variable(
    name = "conv_output",
    dtype = np.float64,
    shape = output.shape,
)

conv_node = gs.Node(
    op = "Conv",
    name = "conv",
    attrs = {
        "kernel_shape": [1, 1],
        "dilations": [1, 1],
        "strides": [1, 1],
        "group": 1,
        "pads": [0, 0, 0, 0],
    },
    inputs = [input_tensor, weights_tensor],
    outputs = [conv_output_tensor],
)

mul_tensor = gs.Constant(
    name = "mul",
    values = np.asarray([1] * channel_out).reshape(batch, channel_out, 1, 1).astype(np.float64),
)

add_tensor = gs.Constant(
    name = "add",
    values = np.asarray([0] * channel_out).reshape(batch, channel_out, 1, 1).astype(np.float64),
)

rqs_output_tensor = gs.Variable(
    name = "rqs_output",
    dtype = np.float64,
    shape = output.shape,
)

rqs_node = gs.Node(
    op = "RequantShift",
    name = "rqs",
    attrs = {
        "div": gs.Constant(name = 'div', values = np.asarray([1], dtype = np.float64)),
        "n_levels": gs.Constant(name = 'n_levels', values = np.asarray([256], dtype = np.float64)),
        "signed": gs.Constant(name = 'signed', values = np.asarray([1], dtype = np.float64)),
    },
    inputs = [conv_output_tensor, mul_tensor, add_tensor],
    outputs = [rqs_output_tensor],
)

graph = gs.Graph(
    nodes = [conv_node, rqs_node],
    inputs = [input_tensor],
    outputs = [rqs_output_tensor],
    name = "test_graph",
)

onnx.save(gs.export_onnx(graph), "network.onnx")
