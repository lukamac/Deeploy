from Deeploy.DeeployTypes import NodeTemplate

template = NodeTemplate("""
// SiLU (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
Silu(${data_in}, ${data_out}, ${size});
END_SINGLE_CORE
""")
