from __future__ import annotations

from collections import defaultdict
from uuid import NAMESPACE_URL, UUID, uuid5


PROJECT_NAMESPACE = uuid5(NAMESPACE_URL, "crypto-fraud-platform")


def deterministic_uuid(kind: str, *parts: object) -> str:
    key = ":".join([kind, *(str(part) for part in parts)])
    return str(uuid5(PROJECT_NAMESPACE, key))


class DeterministicIdFactory:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._counters: defaultdict[str, int] = defaultdict(int)

    def next(self, kind: str) -> str:
        index = self._counters[kind]
        self._counters[kind] += 1
        return deterministic_uuid(kind, self.seed, index)

    def current_index(self, kind: str) -> int:
        return self._counters[kind]


def ensure_uuid(value: str) -> UUID:
    return UUID(value)
