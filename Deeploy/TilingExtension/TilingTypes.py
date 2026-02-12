# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional


@dataclass
class Lifetime:
    """Represents a buffer lifetime"""
    start: int
    duration: int

    def __init__(self, start: int, duration: int) -> None:
        if start < 0:
            raise ValueError(f"Lifetime start should be a positive number (including zero). Received start {start}")
        if duration < 0:
            raise ValueError(f"Duration should be a positive number (including zero). Received duration {duration}")
        self.start = start
        self.duration = duration

    @property
    def end(self) -> int:
        return self.start + self.duration

    def contains(self, timestamp: int) -> bool:
        return self.start <= timestamp and timestamp <= self.start + self.duration

    def overlaps(self, other: "Lifetime") -> bool:
        return self.contains(other.start) or other.contains(self.start)

    def setEnd(self, end: int) -> None:
        if end < self.start:
            raise ValueError(f"End cannot be smaller then the start. Tried to set {self.__repr__()} end to {end}")
        self.duration = end - self.start


@dataclass
class AddressSpace:
    """Represents a buffer's memory address space"""
    base: int
    size: int

    def __init__(self, base: int, size: int) -> None:
        if base < 0:
            raise ValueError(f"Base address should be a positive number (including zero). Received base address {base}")
        if size < 0:
            raise ValueError(f"Size should be a positive number (including zero). Received size {size}")
        self.base = base
        self.size = size

    @property
    def end(self) -> int:
        return self.base + self.size

    def contains(self, address: int) -> bool:
        return self.base <= address and address < self.end

    def overlaps(self, other: "AddressSpace") -> bool:
        return self.contains(other.base) or other.contains(self.base)


@dataclass
class MemoryBlock:
    name: str
    level: str
    lifetime: Lifetime
    addrSpace: Optional[AddressSpace] = None

    def collides(self, other: "MemoryBlock") -> bool:
        if self.addrSpace is None or other.addrSpace is None:
            return False
        return self.lifetime.overlaps(other.lifetime) and self.addrSpace.overlaps(other.addrSpace)
