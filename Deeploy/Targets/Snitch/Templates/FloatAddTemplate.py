# ----------------------------------------------------------------------
#
# File: FloatAddTemplate.py
#
# Last edited: 11.06.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author:
# - Luka Macan, luka.macan@unibo.it, University of Bologna
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from Deeploy.DeeployTypes import NodeTemplate


referenceTemplate = NodeTemplate("""
// Snitch Float Add (Name: ${nodeName}, Op: ${nodeOp})
SnitchFloatAdd(${data_in_1}, ${data_in_2}, ${data_out}, ${size});
""")
