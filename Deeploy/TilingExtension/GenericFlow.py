# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Generic, Set, TypeVar

FlowType = TypeVar("FlowType")
StepType = TypeVar("StepType")


# SCHEREMO: Checkout data-flow analysis (https://en.wikipedia.org/wiki/Data-flow_analysis)
class GenericFlow(Generic[FlowType, StepType]):

    def computeLive(self, live: Set[FlowType], gen: Set[FlowType], kill: Set[FlowType]) -> Set[FlowType]:
        assert gen.isdisjoint(kill), \
                f"ERROR: Generating and killing {FlowType} instance: {gen & kill}"
        assert gen.isdisjoint(live), \
                f"ERROR: Generating an already live {FlowType} instance: {gen & live}"
        assert live.issuperset(kill), \
                f"ERROR: Killing a non-live {FlowType} instance: {kill - live}"
        return (live | gen) - kill

    @abstractmethod
    def computeGen(self, step: StepType) -> Set[FlowType]:
        pass

    @abstractmethod
    def computeKill(self, step: StepType) -> Set[FlowType]:
        pass
