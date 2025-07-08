# ----------------------------------------------------------------------
#
# File: IntrospectiveBinding.py
#
# Last edited: 10.06.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import copy
import types
from typing import Dict, List

import mako.codegen as codegen
from mako.lexer import Lexer
from mako.parsetree import ControlLine, Expression, TemplateNode

from Deeploy.AbstractDataTypes import Pointer, Struct
from Deeploy.DeeployTypes import ExecutionBlock, NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer

_NULL: str = "NULL"


class IntrospectiveCodeTransformationMixIn():

    parseTreeDict: Dict[int, TemplateNode] = {}

    @staticmethod
    def _generateParseTree(template: NodeTemplate) -> TemplateNode:
        return Lexer(template.template._source).parse()

    @staticmethod
    def _reconstructCode(template: NodeTemplate, node: TemplateNode):

        def fixupParseTree(parseTree: TemplateNode) -> TemplateNode:
            nodes = []
            prevLine = 0
            prevPos = 0
            for node in parseTree.nodes:

                newNode = copy.copy(node)
                offset = len(node.source)

                # Expression contain the actual expression + the symbols "${}", i.e. 3 offset symbols
                if isinstance(newNode, Expression):
                    offset += 3

                prevPos = prevPos + offset

                if prevLine != node.lineno:
                    prevPos = node.pos

                newNode.pos = prevPos
                prevLine = node.lineno

                nodes.append(newNode)

            parseTree.nodes = nodes

            return parseTree

        node = fixupParseTree(node)

        temp = template.template
        lexer = Lexer(temp._source)
        source = codegen.compile(
            node,
            temp.uri,
            None,
            default_filters = temp.default_filters,
            buffer_filters = temp.buffer_filters,
            imports = temp.imports,
            future_imports = temp.future_imports,
            source_encoding = lexer.encoding,
            generate_magic_comment = True,
            strict_undefined = temp.strict_undefined,
            enable_loop = temp.enable_loop,
            reserved_names = temp.reserved_names,
        )
        module = types.ModuleType(temp.module_id)
        code = compile(source, temp.module_id, "exec")
        exec(code, module.__dict__, module.__dict__)

        temp._code = code
        temp.module = module
        temp.callable_ = temp.module.render_body
        template.template = temp

    def extractDynamicReferences(self,
                                 ctxt: NetworkContext,
                                 executionBlock: ExecutionBlock = None,
                                 unrollStructs = False,
                                 includeGobalReferences = False):

        makoDynamicReferences = []
        for codeSnippet in executionBlock.codeSnippets:
            template, operatorRepresentation = codeSnippet.template, codeSnippet.operatorRepresentation

            newRefs = self._extractDynamicExpressions(ctxt, operatorRepresentation, template, unrollStructs,
                                                      includeGobalReferences)

            makoDynamicReferences += newRefs

        ret = IntrospectiveCodeTransformationMixIn._fixCtxtOrdering(ctxt, list(dict.fromkeys(makoDynamicReferences)))

        return ret

    @staticmethod
    def _fixCtxtOrdering(ctxt: NetworkContext, nameList: List[str]) -> List[str]:

        orderList = [*ctxt.globalObjects.keys(), *ctxt.localObjects.keys()]
        _nameList = sorted(nameList.copy(), key = lambda key: orderList.index(key))

        return _nameList

    def _extractDynamicExpressions(self,
                                   ctxt: NetworkContext,
                                   operatorRepresentation: OperatorRepresentation,
                                   template: NodeTemplate,
                                   unrollStructs = False,
                                   includeGobalReferences = False):
        codeHash = hash(template.template._source)

        if codeHash in self.parseTreeDict.keys():
            makoParseTree = self.parseTreeDict[codeHash]
        else:
            # Parse the user-provided template
            makoParseTree = IntrospectiveCodeTransformationMixIn._generateParseTree(template)
            self.parseTreeDict[codeHash] = makoParseTree

        # Filter parsing tree for expressions
        makoExpressions = [node.text for node in makoParseTree.nodes if isinstance(node, Expression)]

        # Filter parsing tree for for loops
        makoForLoops = [
            node.text
            for node in makoParseTree.nodes
            if isinstance(node, ControlLine) and node.keyword == 'for' and node.text != 'endfor'
        ]

        # Extract the looped over expression
        def extractLoopedOverExpression(text: str) -> List[str]:
            text += "\n  pass"  # For loop has to have a body for ast.parse to succeed
            tree = ast.parse(text)
            assert len(tree.body) == 1
            node = tree.body[0]
            assert isinstance(node, ast.For)
            forNode = node
            return [node.id for node in ast.walk(forNode.iter) if isinstance(node, ast.Name)]

        for loopText in makoForLoops:
            makoExpressions.extend(extractLoopedOverExpression(loopText))

        # Filter represented expressions
        representedExpressions = [
            operatorRepresentation[expr] for expr in makoExpressions if expr in operatorRepresentation
        ]

        # Flatten represented expressions
        flattenedRepresentedExpressions = []
        for expr in representedExpressions:
            if isinstance(expr, list):
                flattenedRepresentedExpressions.extend(expr)
            else:
                flattenedRepresentedExpressions.append(expr)

        # Filter buffers from expressions
        references = [expr for expr in flattenedRepresentedExpressions if ctxt.is_buffer(expr)]

        if unrollStructs:

            def _unrollStructReferences(val: Struct) -> List[str]:
                assert isinstance(val, Struct)
                # Recursively unroll struct references
                structReferences = []
                for field in val.value.values():
                    if isinstance(field, Struct):
                        structReferences += _unrollStructReferences(field)
                    elif isinstance(field, Pointer) and field.referenceName != _NULL:
                        structReferences.append(field.referenceName)
                return structReferences

            # Unroll local struct references
            for ref in references:
                if hasattr(ctxt.lookup(ref), "structDict"):
                    references += _unrollStructReferences(ctxt.lookup(ref).structDict)

        # Filter expressions for local variables contained in operatorRepresentation
        localReferences = [ref for ref in references if ctxt.is_local(ref)]

        # Filter expressions for global variables contained in operatorRepresentation
        globalReferences = [ref for ref in references if ctxt.is_global(ref)]

        # Filter for dynamically allocated tensors
        dynamicLocalReferences = [ref for ref in localReferences if ctxt.lookup(ref)._deploy]
        dynamicGlobalReferences = [ref for ref in globalReferences if isinstance(ctxt.lookup(ref), VariableBuffer)]

        if includeGobalReferences:
            return dynamicLocalReferences + dynamicGlobalReferences
        else:
            return dynamicLocalReferences
