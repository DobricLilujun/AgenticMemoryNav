"""Small deterministic vector index used as a dependency-free fallback."""

from __future__ import annotations

import hashlib
import re

import numpy as np


class LocalVectorStore:
    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def embed_text(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in re.findall(r"[\w]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            vector[index] += 1.0 if digest[4] % 2 else -1.0
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        return vector.tolist()

    @staticmethod
    def similarity(left: list[float] | None, right: list[float] | None) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        left_array, right_array = np.asarray(left), np.asarray(right)
        denominator = np.linalg.norm(left_array) * np.linalg.norm(right_array)
        return float(np.dot(left_array, right_array) / denominator) if denominator else 0.0
