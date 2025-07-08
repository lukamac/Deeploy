# ----------------------------------------------------------------------
#
# File: typeMapping.py
#
# Last edited: 22.05.2023
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

from collections import namedtuple
from typing import List, Optional, Type

import numpy as np
import numpy.typing as npt

from Deeploy.AbstractDataTypes import BaseType, PointerClass
from Deeploy.CommonExtensions.DataTypes import FloatDataTypes, IntegerDataTypes, float32_t, int8_t, int16_t, int32_t, \
    uint8_t, uint16_t, uint32_t

offsetType = namedtuple("offsetType", ("type", "offset"))


def isInteger(_input: np.array) -> bool:
    if np.abs((_input.astype(int) - _input)).max() > 0.001:
        return False
    return True


def isUnsigned(_input: np.array) -> bool:
    if (_input).min() < 0:
        return False
    return True


def dataWidth(n):
    count = 0
    n = np.abs(int(n - 1))
    while (n > 0):
        count += 1
        n = n >> 8
    ret = 2**(count + 2)
    if ret < 8:
        ret = 8
    return ret


def inferInputType(_input: np.ndarray,
                   signProp: Optional[bool] = None,
                   defaultType = PointerClass(int8_t),
                   defaultOffset = 0) -> List[offsetType]:

    # WIESEP: We cannot do type inference for empty arrays.
    if np.prod(_input.shape) == 0:
        print(f"Warning: Empty input array for type inference for {_input}!")
        return [(defaultType, defaultOffset)]

    if signProp is None:
        signProp = False

    signedPlatformTypes = [_type for _type in IntegerDataTypes if _type.typeMin < 0]

    matchingTypes = []

    # FIXME: this is okay for now (3 distinctions are fine), but there is implicit
    # knowledge encoded in the order of the checks (i.e. first unsigned, signed
    # and then float). It might be good to extract that implicit knowledge into an ordered list.
    if signProp and isUnsigned(_input) and isInteger(_input):
        for _type in sorted(signedPlatformTypes, key = lambda x: x.typeWidth):
            signPropOffset = (2**(_type.typeWidth - 1))
            if _type.checkPromotion(_input - signPropOffset):
                matchingTypes.append(offsetType(PointerClass(_type), signPropOffset))
    elif isInteger(_input):
        for _type in sorted(IntegerDataTypes, key = lambda x: x.typeWidth):
            if _type.checkPromotion(_input):
                matchingTypes.append(offsetType(PointerClass(_type), 0))
    else:
        for _type in sorted(FloatDataTypes, key = lambda x: x.typeWidth):
            if _type.checkPromotion(_input):
                matchingTypes.append(offsetType(PointerClass(_type), 0))

    if matchingTypes == []:
        raise Exception("Could not find a matching type!")

    return matchingTypes


def baseTypeFromName(name: str) -> Type[BaseType]:
    if name == "int8_t":
        return int8_t
    elif name == "uint8_t":
        return uint8_t
    elif name == "int16_t":
        return int16_t
    elif name == "uint16_t":
        return uint16_t
    elif name == "int32_t":
        return int32_t
    elif name == "uint32_t":
        return uint32_t
    elif name == "float32_t":
        return float32_t
    else:
        raise RuntimeError(f"Unrecognized name {name}")


def dtypeFromDeeployType(_ty: Type[BaseType]) -> npt.DTypeLike:
    if _ty == int8_t:
        return np.int8
    elif _ty == uint8_t:
        return np.uint8
    elif _ty == int16_t:
        return np.int16
    elif _ty == uint16_t:
        return np.uint16
    elif _ty == int32_t:
        return np.int32
    elif _ty == uint32_t:
        return np.uint32
    elif _ty == float32_t:
        return np.float32
    else:
        raise RuntimeError(f"Unimplemented conversion for type {_ty.typeName}")
