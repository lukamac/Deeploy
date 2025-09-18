# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import List, Sequence, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer


def coalesce_axes_rowmajor(shape: Sequence[int], axes: Sequence[int]) -> Tuple[List[int], List[int]]:
    """
    Coalesce adjacent iteration axes for a contiguous row-major (C-order) tensor.

    Returns:
      sizes   : merged loop sizes, outer -> inner
      strides : corresponding element strides (distance between consecutive
                items along that merged loop)
    """
    rank = len(shape)
    if rank == 0 or not axes:
        return [], []

    # Build row-major strides in ELEMENTS: stride[i] = Π shape[i+1:]
    strides = [0] * rank
    strides[-1] = 1
    for i in range(rank - 2, -1, -1):
        strides[i] = strides[i + 1] * max(1, shape[i + 1])

    # Normalize + sort unique axes
    norm = []
    for a in axes:
        a = a + rank if a < 0 else a
        if not (0 <= a < rank):
            raise ValueError(f"axis {a} out of range for rank {rank}")
        norm.append(a)
    norm = sorted(set(norm))
    if not norm:
        return [], []

    sizes, out_strides = [], []
    run_start = norm[0]
    run_prev = norm[0]
    last_stride_seen = None

    for a in norm[1:] + [None]:  # sentinel to flush last run
        if a is None or a != run_prev + 1:
            # Close run [run_start..run_prev]
            size = math.prod(shape[run_start:run_prev+1])
            stride = strides[run_start]  # row-major: use last axis' stride
            last_stride_seen = stride
            # skip `size == 1` shapes
            if size > 1:
                sizes.append(int(size))
                out_strides.append(int(stride))
            # Start next run
            run_start = a
        run_prev = a

    # If *everything* was size 1, optionally expose a single dummy loop
    if not sizes:
        # Choose a representative stride; the last run’s last axis is fine.
        # (If caller prefers “no loops”, they can ignore this and treat [] as no-op.)
        rep_stride = 1 if last_stride_seen is None else int(last_stride_seen)
        sizes = [1]
        out_strides = [rep_stride]

    return sizes, out_strides


class ReduceSumTemplate(NodeTemplate):

    def alignToContext(self, ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, OperatorRepresentation, List[str]]:
        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        assert isinstance(data_in, VariableBuffer)
        data_out = ctxt.lookup(operatorRepresentation['data_out'])
        assert isinstance(data_out, VariableBuffer)

        operatorRepresentation['data_out_size'] = math.prod(data_out.shape)

        axes = operatorRepresentation['axes']
        if axes is not None:
            # Make into list
            if isinstance(axes, (int, float)):
                axes = [int(axes)]

            ends, strides = coalesce_axes_rowmajor(data_in.shape, axes)
        else:
            ends, strides = [math.prod(data_in.shape)], [1]

        operatorRepresentation['loop_ends'] = ends
        operatorRepresentation['loop_strides'] = strides

        return ctxt, operatorRepresentation, []


referenceTemplate = ReduceSumTemplate("""
// Float ReduceSum (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
for (uint32_t i = 0; i < ${data_out_size}; i++) {
    float32_t sum = 0.0f;\\

    % for end in loop_ends:
    for (uint32_t i_${loop.index} = 0; i_${loop.index} < ${end}; i_${loop.index}++) {
    % endfor
        uint32_t index = \\
    % for stride in loop_strides:
    i_${loop.index} * ${stride} \\
    % if not loop.last:
    + \\
    % else:
    ;\\
    % endif
    % endfor

    sum += ${data_in}[index];
    % for _ in loop_ends:
    }\\
    % endfor

    ${data_out}[i] = sum;
}
END_SINGLE_CORE
""")
