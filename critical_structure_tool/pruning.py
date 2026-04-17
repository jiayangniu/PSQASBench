from __future__ import annotations

import math
import numpy as np

from .circuit_utils import optimize_circuit, remove_gate
from .types import AblationCandidate, BranchState, GateSpec


def compute_gate_importance(
    gates: list[GateSpec],
    n_qubits: int,
    H: np.ndarray,
    energy_shift: float,
    exact_energy: float,
    protected_gate_signatures: set[str],
    maxiter: int,
    n_restarts: int,
    optimizer: str,
    rotosolve_sweeps: int,
    rng: np.random.Generator,
) -> tuple[float, float, list[AblationCandidate], list[GateSpec]]:
    baseline_energy, baseline_gates = optimize_circuit(
        gates,
        n_qubits,
        H,
        energy_shift,
        maxiter=maxiter,
        n_restarts=n_restarts,
        optimizer=optimizer,
        rotosolve_sweeps=rotosolve_sweeps,
        rng=rng,
    )
    baseline_error_mha = abs(baseline_energy - exact_energy) * 1000.0

    candidates: list[AblationCandidate] = []
    for gate_index, gate in enumerate(baseline_gates):
        remaining, init_angles = remove_gate(baseline_gates, gate_index)
        child_energy, child_gates = optimize_circuit(
            remaining,
            n_qubits,
            H,
            energy_shift,
            init_angles=init_angles,
            maxiter=maxiter,
            n_restarts=n_restarts,
            optimizer=optimizer,
            rotosolve_sweeps=rotosolve_sweeps,
            rng=rng,
        )
        child_error_mha = abs(child_energy - exact_energy) * 1000.0
        candidates.append(
            AblationCandidate(
                gate_index=gate_index,
                gate_label=gate.detailed_signature,
                action_id=gate.action_id,
                delta_error_mha=child_error_mha - baseline_error_mha,
                child_error_mha=child_error_mha,
                child_gates=child_gates,
                is_anchor=gate.signature in protected_gate_signatures,
            )
        )

    candidates.sort(key=lambda c: (c.delta_error_mha, c.child_error_mha))
    return baseline_energy, baseline_error_mha, candidates, baseline_gates


def _gate_key(gate: GateSpec) -> tuple[int, int]:
    return (int(gate.step_index), int(gate.action_id))


def _candidate_rank_cost(delta_error_mha: float, is_anchor: bool, protected_penalty: float) -> float:
    base = abs(float(delta_error_mha))
    if is_anchor:
        return base / max(float(protected_penalty), 1e-9)
    return base


def probabilistic_beam_prune(
    baseline_gates: list[GateSpec],
    baseline_error_mha: float,
    initial_candidates: list[AblationCandidate],
    n_qubits: int,
    H: np.ndarray,
    energy_shift: float,
    exact_energy: float,
    allowed_error_mha: float,
    delta_tolerance_mha: float,
    protected_gate_signatures: set[str],
    beam_width: int,
    branching_factor: int,
    max_prune_steps: int,
    temperature_mha: float,
    protected_penalty: float,
    maxiter: int,
    n_restarts: int,
    optimizer: str,
    rotosolve_sweeps: int,
    prune_budget: int,
    rng: np.random.Generator,
) -> tuple[BranchState, list[BranchState]]:
    root = BranchState(
        gates=baseline_gates,
        error_mha=baseline_error_mha,
        removal_trace=[],
        total_delta_mha=0.0,
    )

    prior_rank_costs = {}
    for cand in initial_candidates:
        if 0 <= cand.gate_index < len(baseline_gates):
            gate = baseline_gates[cand.gate_index]
            prior_rank_costs[_gate_key(gate)] = _candidate_rank_cost(
                cand.delta_error_mha,
                cand.is_anchor,
                protected_penalty,
            )

    evaluations_used = len(initial_candidates)
    beam = [root]
    all_seen_states = [root]

    for _ in range(max_prune_steps):
        if evaluations_used >= prune_budget:
            break
        next_beam: list[BranchState] = []
        expanded_any = False

        for branch in beam:
            ranked_gate_indices = []
            for gate_index, gate in enumerate(branch.gates):
                rank_cost = prior_rank_costs.get(_gate_key(gate), float("inf"))
                ranked_gate_indices.append((rank_cost, gate_index, gate))
            ranked_gate_indices.sort(key=lambda item: (item[0], item[2].step_index, item[2].action_id))

            top_k = max(1, int(branching_factor))
            candidate_indices = [gate_index for _, gate_index, _ in ranked_gate_indices[:top_k]]

            viable_children: list[BranchState] = []
            for gate_index in candidate_indices:
                if evaluations_used >= prune_budget:
                    break
                removed_gate = branch.gates[gate_index]
                remaining, init_angles = remove_gate(branch.gates, gate_index)
                child_energy, child_gates = optimize_circuit(
                    remaining,
                    n_qubits,
                    H,
                    energy_shift,
                    init_angles=init_angles,
                    maxiter=maxiter,
                    n_restarts=n_restarts,
                    optimizer=optimizer,
                    rotosolve_sweeps=rotosolve_sweeps,
                    rng=rng,
                )
                evaluations_used += 1
                child_error_mha = abs(child_energy - exact_energy) * 1000.0
                delta_error_mha = child_error_mha - branch.error_mha
                if child_error_mha > allowed_error_mha or abs(delta_error_mha) > delta_tolerance_mha:
                    continue
                viable_children.append(
                    BranchState(
                        gates=child_gates,
                        error_mha=child_error_mha,
                        removal_trace=branch.removal_trace + [removed_gate.detailed_signature],
                        total_delta_mha=child_error_mha - baseline_error_mha,
                    )
                )

            if not viable_children:
                all_seen_states.append(branch)
                continue

            expanded_any = True
            next_beam.extend(viable_children)

        if not expanded_any:
            break

        dedup: dict[tuple[int, ...], BranchState] = {}
        for branch in next_beam:
            key = branch.structure_signature
            keep = dedup.get(key)
            if keep is None or (len(branch.gates), branch.error_mha, branch.total_delta_mha) < (
                len(keep.gates),
                keep.error_mha,
                keep.total_delta_mha,
            ):
                dedup[key] = branch

        beam = sorted(
            dedup.values(),
            key=lambda b: (len(b.gates), b.error_mha, b.total_delta_mha),
        )[:beam_width]
        all_seen_states.extend(beam)

    best = min(all_seen_states, key=lambda b: (len(b.gates), b.error_mha, b.total_delta_mha))
    return best, sorted(
        all_seen_states,
        key=lambda b: (len(b.gates), b.error_mha, b.total_delta_mha),
    )
