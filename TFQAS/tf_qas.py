"""
tf_qas.py — Two-stage progressive Training-Free QAS algorithm.

Algorithm 1 (He et al., AAAI-24):

  Input:  S (total sampled circuits), R (kept after Stage 1), K (final output)
  Output: top-K circuits ranked by expressibility

  Stage 1 — Path-based proxy (zero-cost):
    1. Randomly sample S circuits from the search space.
    2. Compute path_count for each circuit.
    3. Keep top-R circuits with the highest path counts.

  Stage 2 — Expressibility proxy (precise):
    4. Compute expressibility ε for the R remaining circuits.
    5. Return top-K circuits sorted by ε (highest first).

  Optional Stage 3 — Ground-truth VQE:
    6. Evaluate the top-K circuits with COBYLA VQE.
    7. Return VQEResult for each circuit.
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from .circuit import Circuit
from .expressibility import compute_expressibility
from .search_space import sample_layerwise, sample_random
from .vqe_eval import VQEResult, evaluate_circuit


# ─── Top-level worker (must be picklable for multiprocessing) ──────────────────

def _expr_worker(args):
    """Compute expressibility for one circuit. Top-level so it can be pickled."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from circuit import Circuit
    from expressibility import compute_expressibility
    import numpy as np
    gates, n_qubits, n_samples, n_bins, seed = args
    rng = np.random.default_rng(seed)
    circ = Circuit(n_qubits, gates)
    return compute_expressibility(circ, n_samples=n_samples, n_bins=n_bins, rng=rng)


@dataclass
class TFQASResult:
    circuit: Circuit
    path_count: int
    expressibility: float
    vqe: VQEResult | None = None       # filled in if VQE evaluation is requested

    def __repr__(self):
        vqe_str = (
            f", energy={self.vqe.energy:.6f} Ha"
            f", error={self.vqe.error_mha:.3f} mHa"
            f", chem={'✓' if self.vqe.reached_chem else '✗'}"
        ) if self.vqe else ""
        return (
            f"TFQASResult("
            f"n_cnot={self.circuit.n_cnot}, "
            f"path_count={self.path_count}, "
            f"expr={self.expressibility:.4f}"
            f"{vqe_str})"
        )


def run_tf_qas(
    n_qubits: int,
    n_layers: int,
    S: int = 10000,
    R: int = 1000,
    K: int = 10,
    pipeline: str = 'layerwise',
    n_expr_samples: int = 2000,
    n_bins: int = 75,
    mol: dict | None = None,
    n_vqe_restarts: int = 3,
    cobyla_maxiter: int = 1000,
    seed: int | None = None,
    verbose: bool = True,
    n_workers: int | None = None,
) -> list[TFQASResult]:
    """
    Run the two-stage TF-QAS algorithm.

    Args:
        n_qubits:       number of qubits
        n_layers:       number of (single + two-qubit) layers per circuit
        S:              number of circuits to sample in Stage 1
        R:              number of circuits kept after Stage 1 (R << S)
        K:              number of circuits returned (K <= R)
        pipeline:       'layerwise' or 'random'
        n_expr_samples: number of random parameter pairs for expressibility
        n_bins:         histogram bins for expressibility
        mol:            dict from load_molecule(), if None skips VQE evaluation
        n_vqe_restarts: COBYLA restarts per circuit in VQE evaluation
        cobyla_maxiter: COBYLA max iterations
        seed:           random seed
        verbose:        print progress
        n_workers:      number of parallel processes for Stage 2 (default: all CPUs)

    Returns:
        List of TFQASResult sorted by expressibility (best first, length K).
    """
    rng = np.random.default_rng(seed)
    if n_workers is None:
        n_workers = os.cpu_count() or 1

    _log = print if verbose else lambda *a, **k: None

    # ─────────────────────────── Stage 1: path-based ──────────────────────────
    _log(f"\n[TF-QAS] Stage 1 — sampling {S} circuits ({pipeline}) ...")
    t0 = time.time()

    sampler: Callable = (
        (lambda: sample_layerwise(n_qubits, n_layers, rng))
        if pipeline == 'layerwise'
        else (lambda: sample_random(n_qubits, n_layers * n_qubits, rng))
    )

    circuits = [sampler() for _ in range(S)]
    path_counts = np.array([c.path_count for c in circuits])

    t1 = time.time()
    _log(f"  path counts: min={path_counts.min()}, max={path_counts.max()}, "
         f"mean={path_counts.mean():.1f}  [{t1-t0:.1f}s]")

    # Keep top-R by path count
    top_r_idx = np.argsort(path_counts)[::-1][:R]
    candidate_circuits = [circuits[i] for i in top_r_idx]
    candidate_paths = path_counts[top_r_idx]

    _log(f"  kept top-{R} circuits (min path_count={candidate_paths.min()})")

    # ─────────────────────────── Stage 2: expressibility (parallel) ──────────
    _log(f"\n[TF-QAS] Stage 2 — computing expressibility for {R} circuits "
         f"({n_workers} workers) ...")
    t2 = time.time()

    # Build worker args: each worker gets the gate list + a unique seed
    rng_seq = np.random.SeedSequence(seed)
    worker_seeds = rng_seq.spawn(R)

    worker_args = [
        (circ.gates, n_qubits, n_expr_samples, n_bins, int(ws.generate_state(1)[0]))
        for circ, ws in zip(candidate_circuits, worker_seeds)
    ]

    expr_scores = [None] * R
    completed = 0
    log_every = max(1, R // 10)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(_expr_worker, args): i
            for i, args in enumerate(worker_args)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            expr_scores[idx] = future.result()
            completed += 1
            if verbose and completed % log_every == 0:
                elapsed = time.time() - t2
                _log(f"  {completed}/{R} done  [{elapsed:.1f}s]")

    expr_scores = np.array(expr_scores)
    t3 = time.time()
    _log(f"  expressibility: min={expr_scores.min():.4f}, max={expr_scores.max():.4f}  "
         f"[{t3-t2:.1f}s]")

    # Top-K by expressibility
    top_k_idx = np.argsort(expr_scores)[::-1][:K]

    results = [
        TFQASResult(
            circuit=candidate_circuits[i],
            path_count=int(candidate_paths[i]),
            expressibility=float(expr_scores[i]),
        )
        for i in top_k_idx
    ]

    # ─────────────────────────── Stage 3 (optional): VQE ─────────────────────
    if mol is not None:
        _log(f"\n[TF-QAS] Stage 3 — VQE evaluation of top-{K} circuits ...")
        t4 = time.time()
        for j, result in enumerate(results):
            vqe_res = evaluate_circuit(
                result.circuit, mol,
                n_restarts=n_vqe_restarts,
                cobyla_maxiter=cobyla_maxiter,
                rng=rng,
            )
            result.vqe = vqe_res
            _log(f"  [{j+1}/{K}] {result}")
        t5 = time.time()
        _log(f"  VQE done [{t5-t4:.1f}s]")

    _log(f"\n[TF-QAS] Total time: {time.time()-t0:.1f}s")
    return results
