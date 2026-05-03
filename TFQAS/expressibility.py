"""
expressibility.py — Two training-free proxies for circuit ranking.

Proxy 1 (Stage 1, zero-cost):
    path_count — number of distinct DAG paths (see circuit.py)
    Higher = more structurally complex = more entangling potential.

Proxy 2 (Stage 2, precise):
    expressibility ε(C) = -D_KL(P_circuit(F) ‖ P_Haar(F))
    where F = |<ψ(θ)|ψ(φ)>|² is the state fidelity between two random
    parameter configurations.

    Haar distribution: P_Haar(F) = (N-1)(1-F)^(N-2), N = 2^n_qubits
    [Zyczkowski & Sommers 2005]

    Higher ε (less negative) → closer to Haar → more expressive.
    ε = 0 for a perfect Haar-random ensemble (maximally expressive).

Reference: He et al., "Training-Free Quantum Architecture Search", AAAI-24.
"""

import numpy as np
from .circuit import Circuit


def compute_expressibility(
    circuit: Circuit,
    n_samples: int = 2000,
    n_bins: int = 75,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Compute expressibility ε(C) = -D_KL(P_circuit ‖ P_Haar).

    Args:
        circuit:   Circuit template with n_params parametric gates
        n_samples: number of random (θ, φ) pairs to sample
        n_bins:    number of histogram bins over [0, 1]
        rng:       optional numpy random generator for reproducibility

    Returns:
        Scalar ε ≤ 0.  Higher (less negative) means more expressive.
        Circuits with zero parameters are still evaluated: their fidelity
        distribution collapses to F = 1, which should yield a very poor
        expressibility score rather than 0.0.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = 2 ** circuit.n_qubits

    # ── Sample fidelities ────────────────────────────────────────────────────
    if circuit.n_params == 0:
        # Zero-parameter circuits induce a single deterministic state, so the
        # fidelity distribution is a delta peak at F = 1.
        overlaps = np.ones(n_samples, dtype=np.float64)
    else:
        # Draw 2*n_samples random parameter vectors, run pairs (i, i+n_samples)
        all_params = rng.uniform(-np.pi, np.pi, size=(2 * n_samples, circuit.n_params))

        # Pre-compute all statevectors in a batch
        statevecs = np.array([circuit.statevector(p) for p in all_params])  # (2*S, D)

        sv_a = statevecs[:n_samples]       # (S, D)
        sv_b = statevecs[n_samples:]       # (S, D)

        # |<ψ_a|ψ_b>|²
        overlaps = np.abs(np.einsum('ij,ij->i', sv_a.conj(), sv_b)) ** 2  # (S,)

    # ── Build empirical histogram ─────────────────────────────────────────────
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_width = 1.0 / n_bins

    hist, _ = np.histogram(overlaps, bins=bins)
    # Normalise to a probability mass (sum = 1 over bins)
    p_circuit = hist.astype(float)
    p_circuit += 1e-10            # Laplace smoothing to avoid log(0)
    p_circuit /= p_circuit.sum()

    # ── Haar reference distribution ───────────────────────────────────────────
    bin_centers = (bins[:-1] + bins[1:]) / 2
    p_haar = (N - 1) * (1 - bin_centers) ** (N - 2) * bin_width
    p_haar += 1e-10
    p_haar /= p_haar.sum()

    # ── KL divergence D_KL(P_circuit ‖ P_Haar) ───────────────────────────────
    kl_div = float(np.sum(p_circuit * np.log(p_circuit / p_haar)))

    return -kl_div  # ε = -D_KL, higher is better


def compute_expressibility_gpu(
    circuit: Circuit,
    n_samples: int,
    n_bins: int,
    rng: np.random.Generator,
    bvqe,
) -> float:
    """GPU-batched expressibility using BatchedVQE.batch_statevectors.

    Computes the same ε = -D_KL(P_circuit ‖ P_Haar) as compute_expressibility
    but runs the 2*n_samples statevector simulations as a single batched GPU
    kernel instead of n_samples sequential qulacs CPU calls.

    Only supports gate_set='primitive' (rx, ry, rz, cnot).  Automatically
    falls back to the CPU path for circuits with two-qubit parametric gates.

    Args:
        circuit:  Circuit template (primitive gate set)
        n_samples: number of random (θ, φ) pairs to sample
        n_bins:    histogram bins for the KL divergence estimate
        rng:       numpy random generator (for reproducibility)
        bvqe:      BatchedVQE instance — provides device and batch_statevectors
    """
    import torch

    N = 1 << bvqe.n

    if circuit.n_params == 0:
        overlaps = np.ones(n_samples, dtype=np.float64)
    else:
        try:
            state_tensor = circuit.to_state_tensor()
        except ValueError:
            # paper gate set (xx/yy/zz): fall back to CPU
            return compute_expressibility(circuit, n_samples=n_samples, n_bins=n_bins, rng=rng)

        all_params = rng.uniform(
            -np.pi, np.pi, size=(2 * n_samples, circuit.n_params)
        ).astype(np.float32)
        params_t = torch.tensor(all_params, dtype=torch.float32, device=bvqe.device)

        with torch.no_grad():
            sv_all = bvqe.batch_statevectors(state_tensor, params_t)   # (2S, 2^n)

        sv_a = sv_all[:n_samples]   # (S, 2^n)
        sv_b = sv_all[n_samples:]   # (S, 2^n)
        overlaps = (
            torch.sum(sv_a.conj() * sv_b, dim=1).abs().pow(2)
            .float().cpu().numpy().astype(np.float64)
        )

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_width = 1.0 / n_bins
    hist, _ = np.histogram(overlaps, bins=bins)
    p_circuit = hist.astype(float) + 1e-10
    p_circuit /= p_circuit.sum()
    bin_centers = (bins[:-1] + bins[1:]) / 2
    p_haar = (N - 1) * (1 - bin_centers) ** (N - 2) * bin_width + 1e-10
    p_haar /= p_haar.sum()
    return -float(np.sum(p_circuit * np.log(p_circuit / p_haar)))
