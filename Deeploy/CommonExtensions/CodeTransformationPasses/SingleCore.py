from typing import Tuple
from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, _NoVerbosity, NodeTemplate


class SingleCorePass(CodeTransformationPass):

    def apply(self, ctxt: NetworkContext, executionBlock: ExecutionBlock, name: str, verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        _ = name, verbose
        executionBlock.addLeft(NodeTemplate("BEGIN_SINGLE_CORE\n"), {})
        executionBlock.addRight(NodeTemplate("END_SINGLE_CORE\n"), {})
        return ctxt, executionBlock
