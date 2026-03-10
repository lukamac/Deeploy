#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import sys
from typing import List

from testUtils.deeployRunner import main

if __name__ == "__main__":

    # Define parser setup callback to add Siracusa+Neureka-specific arguments
    def setup_parser(parser):
        parser.add_argument('--cores', type = int, default = 8, help = 'Number of cores (default: 8)\n')
        parser.add_argument('--neureka-wmem', action = 'store_true', help = 'Enable Neureka weight memory\n')
        parser.add_argument('--enable-3x3', action = 'store_true', help = 'Enable 3x3 convolutions\n')

    def add_gen_args(args: argparse.Namespace) -> List[str]:
        gen_args = []
        if args.neureka_wmem:
            gen_args.append('--neureka-wmem')
        if args.enable_3x3:
            gen_args.append('--enable_3x3')
        return gen_args

    sys.exit(
        main(default_platform = "Siracusa_w_neureka",
             default_simulator = "gvsoc",
             tiling_enabled = True,
             parser_setup_callback = setup_parser,
             platform_specific_gen_args = add_gen_args))
