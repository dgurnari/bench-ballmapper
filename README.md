# bench-ballmapper

Benchmark comparing three implementations of the BallMapper algorithm:

| Implementation | Landmark selection | Distance computation | Edge detection | GPU |
|---|---|---|---|---|---|
| **pyBallMapper** | greedy (first-uncovered) | `cdist` + Numba JIT | set intersection (O(L²)) | — |
| **pyBallMapper-BallTree** | greedy (first-uncovered) | scikit-learn BallTree | scipy sparse SpGEMM | — |
| **fast-ballmapper** (BallTree) | greedy (ball-tree radius query) | scikit-learn BallTree | dict-based (O(n) passes) | — |
| **fast-ballmapper** (FAISS) | greedy (FAISS flat index) | FAISS (CPU or GPU) | dict-based (O(n) passes) | optional |
| **GPU-waveMIS** | parallel wavefront MIS | PyTorch GEMM (cuBLAS) | cuSPARSE SpGEMM | CUDA required |

The GPU-waveMIS implementation uses a different algorithm: it selects landmarks in parallel via an index-priority wavefront maximal independent set, rather than the sequential greedy approach. The result is a valid maximal ε-net (verified independently), but it may produce a different landmark set than the CPU implementations.

## Results

Results from a representative run on an Apple M-series laptop (CPU only):

**N-scaling** (eps ≈ 0.78, d=100):

| N | pyBallMapper | fast-ballmapper (BallTree) | fast-ballmapper (FAISS) |
|---|---|---|---|
| 500 | 0.39s | 0.02s | — |
| 1,000 | 0.85s | 0.04s | — |
| 2,000 | 0.85s | 0.04s | — |
| 4,000 | 0.85s | 0.04s | — |
| 8,000 | 0.85s | 0.04s | — |

The CPU implementations produce **identical landmarks, cover sets, and graph edges** on the same input (greedy method, Euclidean metric, default ordering). The GPU implementation uses a different algorithm so it may produce a different (but equally valid) maximal ε-net.

## Repository structure

```
bench-ballmapper/
├── pyproject.toml                 # uv-managed project dependencies
├── benchmarks/
│   ├── __init__.py
│   ├── benchmark_runner.py        # Main benchmark orchestration script
│   ├── gen_synthetic.py           # Deterministic synthetic data generator
│   ├── html_report.py             # HTML report builder (self-contained)
│   ├── gpu_ball_mapper.py         # GPU-waveMIS implementation (standalone)
│   ├── bm_validate.py             # Independent validity checker for GPU results
│   └── wrappers/
│       ├── __init__.py
│       ├── wrapper_pyballmapper.py           # Adapter for pyBallMapper
│       ├── wrapper_pyballmapper_balltree.py  # Adapter for pyBallMapper-BallTree
│       ├── wrapper_fastballmapper.py         # Adapter for fast-ballmapper
│       └── wrapper_gpu.py                    # Adapter for GPU-waveMIS
├── scripts/
│   └── build-pyballmapper-balltree.sh  # Build pyballmapper-balltree from source
├── results/                       # Benchmark output directory (gitignored)
└── .gitignore
```

## Setup

Requires Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/dgurnari/bench-ballmapper
cd bench-ballmapper

# Step 1: Build pyballmapper-balltree from the integrate-balltree branch
./scripts/build-pyballmapper-balltree.sh

# Step 2: Install all dependencies
uv sync

# If running on a CUDA-capable machine, install PyTorch for GPU support:
uv sync --dev
```

The `pyproject.toml` pins the compared packages from their GitHub repositories:

- `pyBallMapper` — [jooyounghahn/pyBallMapper](https://github.com/jooyounghahn/pyBallMapper) (`add-gpu-wavemis-benchmark` branch)
- `pyBallMapper-BallTree` — same repository, `integrate-balltree` branch (packaged as `pyballmapper-balltree`; built locally via `scripts/build-pyballmapper-balltree.sh`)
- `fast-ballmapper` — [jhnrckmnznrs/fast-ballmapper](https://github.com/jhnrckmnznrs/fast-ballmapper)

## Usage

```bash
# Default: pyBallMapper vs fast-ballmapper (BallTree)
uv run python benchmarks/benchmark_runner.py

# All CPU implementations
uv run python benchmarks/benchmark_runner.py --impl pyballmapper pyballmapper-balltree fast-balltree fast-faiss

# Including GPU (when CUDA is available)
uv run python benchmarks/benchmark_runner.py --impl pyballmapper pyballmapper-balltree fast-balltree gpu

# Larger dataset sizes
uv run python benchmarks/benchmark_runner.py --ns 500 1000 2000 4000 8000

# More repetitions for tighter statistics
uv run python benchmarks/benchmark_runner.py --reps 5

# Custom output directory
uv run python benchmarks/benchmark_runner.py --out results/run1
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--ns` | `500 1000 2000 4000 8000` | Dataset sizes for N-scaling test |
| `--eps-list` | auto-calibrated | Epsilon values for eps-scaling test |
| `--scaling-eps` | auto-calibrated | Fixed eps for N-scaling |
| `--scaling-n` | 2000 | Fixed N for eps-scaling |
| `--d` | 100 | Number of features for synthetic data |
| `--reps` | 3 | Repetitions per timing |
| `--impl` | `pyballmapper fast-balltree` | Implementations to benchmark (e.g. `pyballmapper pyballmapper-balltree fast-balltree fast-faiss gpu`) |
| `--gpu-device` | `cuda:0` | CUDA device for GPU impl |
| `--out` | `.` | Output directory |

## Output

The script produces two files in the output directory:

- **`results.json`** — Raw per-cell timings, memory, and metadata
- **`report.html`** — Self-contained report with tables and plots (opens in any browser)

## Benchmark methodology

1. **Correctness check**: On a small subset (N=500, 1000, 2000), all implementations are compared for identical landmarks, cover sets, and graph edges.
2. **N-scaling**: Dataset size increases at fixed ε. Tests how each implementation scales with more points.
3. **ε-scaling**: ε decreases at fixed N. Tests how each implementation handles varying ball radii.
4. **Metrics**: Wall-clock time (mean ± std over repeats) and peak RSS memory (via `tracemalloc`).
5. **Data**: Synthetic Gaussian mixture with 100 features, min-max normalised to [0, 1].
