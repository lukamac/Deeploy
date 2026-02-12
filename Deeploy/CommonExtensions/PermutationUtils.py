# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Sequence, TypeVar


# Permute (0,1,2,3,...,N-2,N-1) -> (0,1,2,3,...,N-1,N-2)
def _swapLastTwoDimsPermutation(N: int) -> List[int]:
    assert N >= 2, "N needs to be larger then 2"
    return [*range(N - 2), N - 1, N - 2]


# Permute channels first <-> channels last:
#   (*<batch dims>, ch, *<spatial dims>) <-> (*<batch dims>, *<spatial dims>, ch)
def _transformLayoutPermutation(dims: int, spatialDims: int, targetChannelsFirst: bool) -> List[int]:
    batchDims = dims - spatialDims - 1
    if targetChannelsFirst:
        ch = dims - 1
        nonBatchPerm = [ch, *range(batchDims, ch)]
    else:
        ch = batchDims
        nonBatchPerm = [*range(ch + 1, dims), ch]
    return list(range(batchDims)) + nonBatchPerm


# Calculate permutation q = p^(-1) s.t. q(p(i)) = i
def _invertPermutation(permutation: Sequence[int]) -> List[int]:
    return [permutation.index(i) for i in range(len(permutation))]


T = TypeVar('T')


def _permute(_list: Sequence[T], permutation: Sequence[int]) -> List[T]:
    assert len(_list) == len(permutation), "Permuted list and permutation must have equal length!"
    return [_list[i] for i in permutation]
