from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunContext:
    run_dir: Path
    method: str
    molecule: str
    exp_group: str
    config_name: str
    seed_name: str
    accept_err_ha: float | None
    analysis_save_threshold_ha: float | None
    num_qubits: int
    connectivity: str
    action_dict: dict[int, list[int]]
    mol_path: Path
    train_optimizer: str
    train_cobyla_maxiter: int | None = None
    train_rotosolve_sweeps: int | None = None


@dataclass
class SnapshotRecord:
    run: RunContext
    episode_index: int
    total_episodes: int
    snapshot_index: int
    event_step: int
    event_error_ha: float
    event_error_mha: float
    bucket: str
    action_ids_prefix: list[int]
    last_action_id: int
    last_action_token: str
    snapshot: dict[str, Any] = field(default_factory=dict)
    gates_direct: list[dict] | None = field(default=None)


@dataclass
class GateSpec:
    gate_type: str
    action_id: int
    step_index: int
    ctrl: int | None = None
    targ: int | None = None
    q: int | None = None
    axis: int | None = None
    angle: float = 0.0

    @property
    def signature(self) -> str:
        if self.gate_type == "cnot":
            return f"CNOT({self.ctrl}->{self.targ})"
        axis_name = {1: "RX", 2: "RY", 3: "RZ"}.get(self.axis, f"R{self.axis}")
        return f"{axis_name}(q={self.q})"

    @property
    def detailed_signature(self) -> str:
        if self.gate_type == "cnot":
            return f"{self.signature}@step{self.step_index}[a{self.action_id}]"
        return f"{self.signature}@step{self.step_index}[a{self.action_id},θ={self.angle:+.3f}]"


@dataclass
class AblationCandidate:
    gate_index: int
    gate_label: str
    action_id: int
    delta_error_mha: float
    child_error_mha: float
    child_gates: list[GateSpec]
    is_anchor: bool


@dataclass
class BranchState:
    gates: list[GateSpec]
    error_mha: float
    removal_trace: list[str] = field(default_factory=list)
    total_delta_mha: float = 0.0

    @property
    def structure_signature(self) -> tuple[int, ...]:
        return tuple(g.action_id for g in self.gates)
