from __future__ import annotations

import networkx as nx
import numpy as np

# Import from sibling module — requires the benchmarks dir to be on sys.path
from gpu_ball_mapper import GpuBallMapper  # noqa: E402


class GpuBallMapperWrapper:
    def __init__(self, X: np.ndarray, eps: float, device: str = "cuda:0") -> None:
        self.X = X
        self.eps = eps
        self.device = device

    def build(self) -> tuple[list[int], dict[int, list[int]], nx.Graph]:
        bm = GpuBallMapper(X=self.X, eps=self.eps, device=self.device, verbose=False)
        landmarks = [int(bm.Graph.nodes[n]["landmark"]) for n in bm.Graph.nodes]
        cover = {
            int(k): [int(p) for p in v] for k, v in bm.points_covered_by_landmarks.items()
        }
        graph = bm.Graph
        return landmarks, cover, graph

    def __repr__(self) -> str:
        return "GPU-waveMIS (PyTorch + cuBLAS + cuSPARSE)"
