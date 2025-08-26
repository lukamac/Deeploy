# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class FloatConv2dTemplate(NodeTemplate):

    def alignToContext(
            self, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, OperatorRepresentation, List[str]]:
        if "bias" in operatorRepresentation and "has_bias" in operatorRepresentation:
            bias = operatorRepresentation["bias"]
            has_bias = operatorRepresentation["has_bias"]
            assert (has_bias == 1 and bias != "NULL") or (
                has_bias == 0 and bias == "NULL"), f"Unsupported combination bias: {bias} with has_bias: {has_bias}"
        elif "bias" in operatorRepresentation and "has_bias" not in operatorRepresentation:
            bias = operatorRepresentation["bias"]
            operatorRepresentation["has_bias"] = 0 if bias == "NULL" else 1
        elif "bias" not in operatorRepresentation and "has_bias" in operatorRepresentation:
            has_bias = operatorRepresentation["has_bias"]
            assert has_bias == 0, f"has_bias should be 0 if there is not bias defined in the operator representation. Received has_bias: {has_bias}"
            operatorRepresentation["bias"] = "NULL"
        elif "bias" not in operatorRepresentation and "has_bias" not in operatorRepresentation:
            operatorRepresentation["bias"] = "NULL"
            operatorRepresentation["has_bias"] = 0

        nodeName = operatorRepresentation["nodeName"]
        data_in = operatorRepresentation["data_in"]
        data_out = operatorRepresentation["data_out"]

        ref_prefix = nodeName
        if ref_prefix[0].isdigit():
            ref_prefix = "_" + ref_prefix

        assert "data_in_ref" not in operatorRepresentation
        operatorRepresentation["data_in_ref"] = f"{ref_prefix}_{data_in}_ref"

        assert "data_out_ref" not in operatorRepresentation
        operatorRepresentation["data_out_ref"] = f"{ref_prefix}_{data_out}_ref"

        return ctxt, operatorRepresentation, []


reference2DTemplate = FloatConv2dTemplate("""
<%
batchOffsetIn = ch_im_in * dim_im_in_x * dim_im_in_y
batchOffsetOut = ch_im_out * dim_im_out_x * dim_im_out_y
%>

// 2D FP Conv (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    ${data_in_type.typeName} ${data_in_ref} = ${data_in};
    ${data_out_type.typeName} ${data_out_ref} = ${data_out};

    for (uint32_t n=0; n<${batch}; ++n) {
        Conv2d_fp${data_in_type.referencedType.typeWidth}_fp${weight_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}_NCHW(
            ${data_in_ref}, ${ch_im_in}, ${dim_im_in_x}, ${dim_im_in_y},
            ${weight}, 0, ${ch_im_out}, ${dim_kernel_x}, ${dim_kernel_y},
            ${stride_x}, ${stride_y},
            ${bias},
            ${has_bias},
            ${data_out_ref}
        );
        ${data_in_ref} += ${batchOffsetIn};
        ${data_out_ref} += ${batchOffsetOut};
    }
END_SINGLE_CORE
""")
