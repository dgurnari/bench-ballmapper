from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from fast_ballmapper import build_mapper, compute_landmarks


class FastBallMapperWrapper:
    def __init__(self, X: np.ndarray, eps: float, method: str = "ball_tree") -> None:
        self.X = X
        self.eps = eps
        self.method = method

    def build(self) -> tuple[list[int], dict[int, list[int]], nx.Graph]:
        landmarks, cover = compute_landmarks(
            self.X,
            eps=self.eps,
            method=self.method,
            metric="euclidean",
        )
        graph = build_mapper(cover)
        cover_dict = {i: [int(p) for p in arr] for i, arr in enumerate(cover)}
        return landmarks, cover_dict, graph

    def __repr__(self) -> str:
        return f"fast-ballmapper (greedy, {self.method})"
