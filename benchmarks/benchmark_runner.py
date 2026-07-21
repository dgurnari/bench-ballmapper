from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import tracemalloc
from collections.abc import Callable
from typing import Any

import matplotlib
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gen_synthetic as gs
from html_report import Report
from wrappers.wrapper_fastballmapper import FastBallMapperWrapper
from wrappers.wrapper_pyballmapper import PyBallMapperWrapper
from wrappers.wrapper_pyballmapper_balltree import PyBallMapperBallTreeWrapper

Wrapper = PyBallMapperWrapper | FastBallMapperWrapper | PyBallMapperBallTreeWrapper

# GPU is optional — only imported if explicitly requested and torch+cuda available
_HAS_CUDA = False
GpuBallMapperWrapper = None
try:
    import torch

    _HAS_CUDA = torch.cuda.is_available()
    if _HAS_CUDA:
        from wrappers.wrapper_gpu import GpuBallMapperWrapper
except ImportError:
    pass


# ── correctness helpers ───────────────────────────────────────────────────────
def _edge_set(graph) -> set[frozenset[int]]:
    return set(frozenset((u, v)) for u, v in graph.edges)


def _landmark_ids_from_graph(graph) -> list[int]:
    return [int(graph.nodes[n]["landmark"]) for n in graph.nodes]


def _cover_from_graph(graph) -> dict[int, list[int]]:
    return {int(k): [int(p) for p in graph.nodes[k]["points covered"]] for k in graph.nodes}


def _graphs_match_from_results(la, ca, ga, lb, cb, gb) -> bool:
    return (
        la == lb
        and {k: frozenset(v) for k, v in ca.items()} == {k: frozenset(v) for k, v in cb.items()}
        and _edge_set(ga) == _edge_set(gb)
    )


# ── timed build ───────────────────────────────────────────────────────────────
def _timed_build(
    wrapper_factory: Callable[[], Wrapper],
    reps: int,
) -> tuple[
    list[float],
    list[float],
]:
    ts: list[float] = []
    peak_mbs: list[float] = []
    for i in range(reps):
        tracemalloc.start()
        t0 = time.perf_counter()
        w = wrapper_factory()
        _, _, gr = w.build()
        elapsed = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        ts.append(elapsed)
        peak_mbs.append(peak_bytes / (1024 * 1024))
    return ts, peak_mbs


# ── N-scaling ─────────────────────────────────────────────────────────────────
def run_n_scaling(
    wrappers: dict[str, Callable[..., Wrapper]],
    ns: list[int],
    eps: float,
    reps: int,
    d: int,
    log: Callable[..., Any] = print,
) -> list[dict]:
    rows: list[dict] = []
    for name, factory in wrappers.items():
        log(f"  {name}  eps={eps:.4f}", flush=True)
        for n in ns:
            X = gs.make_highd(n, d=d, seed=n)
            local_factory: Callable[[], Wrapper] = lambda X=X, factory=factory: factory(X, eps)
            all_times, all_mems = _timed_build(local_factory, reps)
            time_arr = np.array(all_times)
            mem_arr = np.array(all_mems)
            w = factory(X, eps)
            _, _, gr = w.build()
            L = gr.number_of_nodes()
            E = gr.number_of_edges()
            row = {
                "wrapper": name,
                "N": n,
                "eps": eps,
                "L": L,
                "E": E,
                "time_mean": float(time_arr.mean()),
                "time_std": float(time_arr.std()),
                "peak_rss_mean_mb": float(mem_arr.mean()),
                "peak_rss_std_mb": float(mem_arr.std()),
                "times": all_times,
                "peak_rss_mb": all_mems,
            }
            rows.append(row)
            log(
                f"    N={n:>6d}  L={L:>5d}  E={E:>5d}  "
                f"t={row['time_mean']:.3f}+/-{row['time_std']:.3f}s  "
                f"mem={row['peak_rss_mean_mb']:.1f}+/-{row['peak_rss_std_mb']:.1f}MB",
                flush=True,
            )
    return rows


# ── eps-scaling ───────────────────────────────────────────────────────────────
def run_eps_scaling(
    wrappers: dict[str, Callable[..., Wrapper]],
    eps_list: list[float],
    n: int,
    reps: int,
    d: int,
    log: Callable[..., Any] = print,
) -> list[dict]:
    rows: list[dict] = []
    X = gs.make_highd(n, d=d, seed=42)
    for name, factory in wrappers.items():
        log(f"  {name}  N={n}", flush=True)
        for eps in eps_list:
            local_factory: Callable[[], Wrapper] = lambda X=X, eps=eps, factory=factory: factory(X, eps)
            all_times, all_mems = _timed_build(local_factory, reps)
            time_arr = np.array(all_times)
            mem_arr = np.array(all_mems)
            w = factory(X, eps)
            _, _, gr = w.build()
            L = gr.number_of_nodes()
            E = gr.number_of_edges()
            row = {
                "wrapper": name,
                "N": n,
                "eps": eps,
                "L": L,
                "E": E,
                "time_mean": float(time_arr.mean()),
                "time_std": float(time_arr.std()),
                "peak_rss_mean_mb": float(mem_arr.mean()),
                "peak_rss_std_mb": float(mem_arr.std()),
                "times": all_times,
                "peak_rss_mb": all_mems,
            }
            rows.append(row)
            log(
                f"    eps={eps:.4f}  L={L:>5d}  E={E:>5d}  "
                f"t={row['time_mean']:.3f}+/-{row['time_std']:.3f}s  "
                f"mem={row['peak_rss_mean_mb']:.1f}+/-{row['peak_rss_std_mb']:.1f}MB",
                flush=True,
            )
    return rows


# ── correctness check between implementations ─────────────────────────────────
def run_correctness_check(wrappers, ns, eps, d, log=print):
    log("Checking correctness equivalence between implementations...", flush=True)
    results = []
    for n in ns:
        X = gs.make_highd(n, d=d, seed=n)
        builds = {}
        for name, factory in wrappers.items():
            w = factory(X, eps)
            lm, cv, gr = w.build()
            builds[name] = (lm, cv, gr)

        ref_name = list(builds.keys())[0]
        ref_lm, ref_cv, ref_gr = builds[ref_name]
        for name, (lm, cv, gr) in builds.items():
            if name == ref_name:
                continue
            ok = _graphs_match_from_results(ref_lm, ref_cv, ref_gr, lm, cv, gr)
            results.append({"N": n, "ref": ref_name, "target": name, "match": ok})
            status = "✓" if ok else "✗"
            log(f"  N={n:>6d}  {ref_name} == {name}: {status}", flush=True)
    return results


# ── plots ─────────────────────────────────────────────────────────────────────
_COLORS = {
    "pyballmapper (greedy, cdist+numba)": "#2563eb",
    "pyballmapper-balltree (BallTree, scipy SpGEMM)": "#7c3aed",
    "fast-ballmapper (greedy, ball_tree)": "#dc2626",
    "fast-ballmapper (greedy, faiss)": "#059669",
    "GPU-waveMIS (PyTorch + cuBLAS + cuSPARSE)": "#f59e0b",
}
_MARKERS = {
    "pyballmapper (greedy, cdist+numba)": "o",
    "pyballmapper-balltree (BallTree, scipy SpGEMM)": "P",
    "fast-ballmapper (greedy, ball_tree)": "s",
    "fast-ballmapper (greedy, faiss)": "D",
    "GPU-waveMIS (PyTorch + cuBLAS + cuSPARSE)": "^",
}


def _apply_ax_style(ax, xlabel, ylabel):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(ax.get_title(), fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, which="major", ls="-", color="#e8e8e8", alpha=0.7)
    ax.grid(True, which="minor", ls="--", color="#f0f0f0", alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9, fontsize=10)


def _plot_method(ax, xs, ys, yerr, label):
    color = _COLORS.get(label, "#7c3aed")
    marker = _MARKERS.get(label, "P")
    ax.plot(xs, ys, marker=marker, color=color, lw=2.5, markersize=8, markeredgecolor="white", markeredgewidth=0.8, label=label, zorder=3)
    ax.fill_between(xs, [y - e for y, e in zip(ys, yerr)], [y + e for y, e in zip(ys, yerr)], color=color, alpha=0.15, zorder=2)


def _n_scaling_figure(rows, eps):
    names = sorted({r["wrapper"] for r in rows})
    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(14, 5.5))
    for name in names:
        subset = [r for r in rows if r["wrapper"] == name]
        ns = [r["N"] for r in subset]
        ts = [r["time_mean"] for r in subset]
        ts_err = [r["time_std"] for r in subset]
        mems = [r["peak_rss_mean_mb"] for r in subset]
        mem_err = [r["peak_rss_std_mb"] for r in subset]
        _plot_method(ax_time, ns, ts, ts_err, name)
        _plot_method(ax_mem, ns, mems, mem_err, name)
    _apply_ax_style(ax_time, "N (points)", "build time (s)")
    _apply_ax_style(ax_mem, "N (points)", "peak RSS (MB)")
    ax_time.set_title(f"Build time vs N (eps={eps:.4f})", fontsize=13, fontweight="bold", pad=10)
    ax_mem.set_title(f"Peak memory vs N (eps={eps:.4f})", fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout(pad=1.5)
    return fig


def _eps_scaling_figure(rows, n):
    names = sorted({r["wrapper"] for r in rows})
    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(14, 5.5))
    for name in names:
        subset = [r for r in rows if r["wrapper"] == name]
        eps_vals = [r["eps"] for r in subset]
        ts = [r["time_mean"] for r in subset]
        ts_err = [r["time_std"] for r in subset]
        mems = [r["peak_rss_mean_mb"] for r in subset]
        mem_err = [r["peak_rss_std_mb"] for r in subset]
        _plot_method(ax_time, eps_vals, ts, ts_err, name)
        _plot_method(ax_mem, eps_vals, mems, mem_err, name)
    _apply_ax_style(ax_time, "eps (ball radius)", "build time (s)")
    _apply_ax_style(ax_mem, "eps (ball radius)", "peak RSS (MB)")
    ax_time.set_title(f"Build time vs eps (N={n})", fontsize=13, fontweight="bold", pad=10)
    ax_mem.set_title(f"Peak memory vs eps (N={n})", fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout(pad=1.5)
    return fig


def _landmarks_figure(rows, eps):
    names = sorted({r["wrapper"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name in names:
        subset = [r for r in rows if r["wrapper"] == name]
        ns = [r["N"] for r in subset]
        landmarks = [r["L"] for r in subset]
        color = _COLORS.get(name, "#7c3aed")
        marker = _MARKERS.get(name, "P")
        ax.plot(ns, landmarks, marker=marker, color=color, lw=2.5, markersize=8, markeredgecolor="white", markeredgewidth=0.8, label=name, zorder=3)
    _apply_ax_style(ax, "N (points)", "number of landmarks")
    ax.set_title(f"Number of landmarks vs N (eps={eps:.4f})", fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout(pad=1.5)
    return fig


# ── report ────────────────────────────────────────────────────────────────────
def build_report(
    n_rows: list[dict],
    eps_rows: list[dict],
    correctness_rows: list[dict],
    meta: dict,
    out_path: str,
) -> str:
    rep = Report(
        "BallMapper benchmark: pyBallMapper vs fast-ballmapper vs GPU-waveMIS",
        subtitle=f"host={meta['host']} · reps={meta['reps']} · d={meta['d']} · {meta['timestamp']}",
    )

    # ── Correctness ──
    rep.h2("Correctness equivalence")
    if correctness_rows:
        all_ok = all(r["match"] for r in correctness_rows)
        rep.callout(
            "All implementations produce <b>identical</b> landmarks, cover sets, and graph edges on shared inputs."
            if all_ok
            else "Some implementations <b>diverge</b> — see the table.",
            kind="good" if all_ok else "warn",
        )
        headers = ["N", "reference", "target", "match"]
        table_rows = [[str(r["N"]), r["ref"], r["target"], "✓" if r["match"] else "✗"] for r in correctness_rows]
        rep.table(headers, table_rows)
    else:
        rep.p("Correctness check not run (e.g. GPU must use different landmark sets by design).")

    # ── N-scaling ──
    rep.h2("Test 1 — N-scaling at fixed eps")
    rep.p(f"Eps = <code>{meta['scaling_eps']:.4f}</code>.")
    headers = ["wrapper", "N", "L", "E", "time (s)", "peak RSS (MB)"]
    table_rows = []
    for r in n_rows:
        table_rows.append([r["wrapper"], f"{r['N']:,}", str(r["L"]), str(r["E"]), f"{r['time_mean']:.3f} +/- {r['time_std']:.3f}", f"{r['peak_rss_mean_mb']:.1f} +/- {r['peak_rss_std_mb']:.1f}"])
    rep.table(headers, table_rows)
    rep.figure(_n_scaling_figure(n_rows, meta["scaling_eps"]), caption="Left: build time vs N (log-log). Right: peak RSS vs N (log-log).")
    rep.figure(_landmarks_figure(n_rows, meta["scaling_eps"]), caption="Number of landmarks vs N (log-log).")

    # ── eps-scaling ──
    rep.h2("Test 2 — eps-scaling at fixed N")
    rep.p(f"N = <code>{meta['scaling_n']:,}</code>.")
    headers = ["wrapper", "eps", "L", "E", "time (s)", "peak RSS (MB)"]
    table_rows = []
    for r in eps_rows:
        table_rows.append([r["wrapper"], f"{r['eps']:.4f}", str(r["L"]), str(r["E"]), f"{r['time_mean']:.3f} +/- {r['time_std']:.3f}", f"{r['peak_rss_mean_mb']:.1f} +/- {r['peak_rss_std_mb']:.1f}"])
    rep.table(headers, table_rows)
    rep.figure(_eps_scaling_figure(eps_rows, meta["scaling_n"]), caption="Left: build time vs eps (log-log). Right: peak RSS vs eps (log-log).")

    # ── Notes ──
    rep.h2("Notes")
    notes = [
        "Euclidean metric. Times and memory reported as mean +/- std over the repeats.",
        "Peak RSS measured via <code>tracemalloc</code> (tracks Python allocations).",
        "Synthetic data: Gaussian mixture, min-max normalised to [0, 1].",
        "pyBallMapper uses <code>cdist</code> + Numba JIT for distances and set intersection for edges.",
        "fast-ballmapper uses scikit-learn BallTree for distances and a dict-based edge construction.",
        "GPU-waveMIS uses PyTorch GEMM (cuBLAS) for distances + wavefront MIS for landmark selection.",
    ]
    if not meta.get("gpu_available"):
        notes.append("GPU-waveMIS was <b>not available</b> on this host (no CUDA or no PyTorch).")
    rep.html("<ul>" + "".join(f"<li>{n}</li>" for n in notes) + "</ul>")

    return rep.save(out_path)


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    globals()["plt"] = plt

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d", type=int, default=100, help="features for synthetic data")
    ap.add_argument("--ns", type=int, nargs="+", default=[500, 1000, 2000, 4000, 8000], help="dataset sizes for N-scaling test")
    ap.add_argument("--eps-list", type=float, nargs="+", default=None, help="eps values for eps-scaling test")
    ap.add_argument("--scaling-eps", type=float, default=None, help="eps for N-scaling test")
    ap.add_argument("--scaling-n", type=int, default=None, help="N for eps-scaling test")
    ap.add_argument("--reps", type=int, default=3, help="repetitions per timing")
    ap.add_argument("--out", default=".", help="output directory")
    ap.add_argument(
        "--impl",
        nargs="+",
        choices=["pyballmapper", "pyballmapper-balltree", "fast-balltree", "fast-faiss", "gpu"],
        default=["pyballmapper", "fast-balltree"],
        help="implementations to benchmark",
    )
    ap.add_argument("--gpu-device", type=str, default="cuda:0", help="CUDA device for GPU impl")
    args = ap.parse_args()

    d = args.d
    ref_full = gs.default_reference(n=max(args.ns), d=d)

    if args.scaling_eps is None:
        cal_eps = gs.calibrate_eps(ref_full[: min(4000, ref_full.shape[0])])
        scaling_eps = cal_eps[1]
    else:
        scaling_eps = args.scaling_eps

    if args.eps_list is not None:
        eps_list = args.eps_list
    else:
        eps_list = gs.calibrate_eps(ref_full[: min(4000, ref_full.shape[0])])

    scaling_n = args.scaling_n or min(2000, max(args.ns))

    wrappers: dict[str, Callable[..., Wrapper]] = {}
    if "pyballmapper" in args.impl:
        wrappers["pyballmapper (greedy, cdist+numba)"] = lambda X, eps: PyBallMapperWrapper(X, eps)
    if "pyballmapper-balltree" in args.impl:
        wrappers["pyballmapper-balltree (BallTree, scipy SpGEMM)"] = lambda X, eps: PyBallMapperBallTreeWrapper(X, eps)
    if "fast-balltree" in args.impl:
        wrappers["fast-ballmapper (greedy, ball_tree)"] = lambda X, eps: FastBallMapperWrapper(X, eps, method="ball_tree")
    if "fast-faiss" in args.impl:
        wrappers["fast-ballmapper (greedy, faiss)"] = lambda X, eps: FastBallMapperWrapper(X, eps, method="faiss")
    if "gpu" in args.impl:
        if GpuBallMapperWrapper is None:
            print("WARNING: GPU impl requested but torch+cuda not available. Skipping GPU.", flush=True)
        else:
            dev = args.gpu_device
            wrappers["GPU-waveMIS (PyTorch + cuBLAS + cuSPARSE)"] = lambda X, eps, dev=dev: GpuBallMapperWrapper(X, eps, device=dev)

    print(f"benchmark_ballmapper | host={socket.gethostname().split('.')[0]} reps={args.reps} d={d}", flush=True)
    print(f"  implementations: {list(wrappers.keys())}", flush=True)
    print(f"  scaling_eps={scaling_eps:.4f}  scaling_n={scaling_n:,}", flush=True)

    # Correctness check (skip GPU — it uses a different landmark set by design)
    correctness_wrappers = {k: v for k, v in wrappers.items() if "GPU" not in k}
    if correctness_wrappers:
        print("\n=== Correctness check ===", flush=True)
        correctness_rows = run_correctness_check(correctness_wrappers, [500, 1000, 2000], scaling_eps, d)
    else:
        correctness_rows = []

    print("\n=== Test 1: N-scaling ===", flush=True)
    n_rows = run_n_scaling(wrappers, args.ns, scaling_eps, args.reps, d)

    print("\n=== Test 2: eps-scaling ===", flush=True)
    eps_rows = run_eps_scaling(wrappers, eps_list, scaling_n, args.reps, d)

    meta = {
        "host": socket.gethostname().split(".")[0],
        "reps": args.reps,
        "d": d,
        "methods": list(wrappers.keys()),
        "scaling_eps": scaling_eps,
        "scaling_n": scaling_n,
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "gpu_available": _HAS_CUDA,
    }

    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "results.json")
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "n_scaling": n_rows, "eps_scaling": eps_rows, "correctness": correctness_rows}, fh, indent=1)
    print(f"\nwrote {results_path}", flush=True)

    report_path = build_report(n_rows, eps_rows, correctness_rows, meta, os.path.join(args.out, "report.html"))
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
