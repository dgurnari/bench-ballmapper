from __future__ import annotations

import networkx as nx
import numpy as np

from pyballmapper import BallMapper


class PyBallMapperBallTreeWrapper:
    def __init__(self, X: np.ndarray, eps: float) -> None:
        self.X = X
        self.eps = eps

    def build(self) -> tuple[list[int], dict[int, list[int]], nx.Graph]:
        bm = BallMapper(X=self.X, eps=self.eps, method="balltree", verbose=False)
        landmarks = [int(bm.Graph.nodes[n]["landmark"]) for n in bm.Graph.nodes]
        cover = {
            int(k): [int(p) for p in v] for k, v in bm.points_covered_by_landmarks.items()
        }
        graph = bm.Graph
        return landmarks, cover, graph

    def __repr__(self) -> str:
        return "pyballmapper-balltree (BallTree, scipy SpGEMM)"
