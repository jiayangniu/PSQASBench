"""
GQERunner — Generative Quantum Eigensolver runner for PSQASBench.

Wraps Q-GESolver (Nakaji et al. 2024, GPT-based circuit generation) as a
BaseRunner so it can be benchmarked alongside CRLQAS / HyRLQAS / QuantumDARTS.

Algorithm:
    1. Build PennyLane Hamiltonian + HF state + UCC operator pool from .npz.
    2. Sample ``train_size`` random operator sequences; compute prefix energies
       with qml.Snapshot (one circuit call per sequence).
    3. Train a GPTQE model: cumsum(logits) ≈ prefix energies (MSE loss).
    4. Every ``eval_every`` epochs: generate ``eval_n_sequences`` sequences,
       evaluate their true energies, record the best one.
    5. Write run_meta.txt, config_used.cfg, episode_traces.txt.

Unlike RL runners, GQERunner does NOT use CircuitEnv.  The ``env`` parameter
in greedy_episode / stochastic_episode is accepted for interface compatibility
but is unused.

Config file sections:
    [env]
        num_qubits        = 6
        accept_err        = 0.0016
        active_electrons  = 2       # electrons in active space
        active_orbitals   = 3       # optional; defaults to n_qubits // 2

    [problem]
        mol_file = L2_LiH_Equil_6q_...npz

    [model]
        seq_len           = 10
        train_size        = 2048
        n_layer           = 6
        n_head            = 8
        n_embd            = 256

    [general]
        epochs            = 10000
        eval_every        = 100
        eval_n_sequences  = 500
        lr                = 0.0003
"""

from __future__ import annotations

import configparser
import sys
import time
from pathlib import Path

import numpy as np
import torch

from RLQAS.base_runner import BaseRunner

# Path to sibling Q-GESolver repo — resolved at import time.
_QGES_DIR = Path(__file__).resolve().parent.parent.parent / "Q-GESolver"


def _import_qges():
    """Add Q-GESolver to sys.path and return (GPTQE, GPTConfig,
    get_subsequence_energies, circuit_resources)."""
    if str(_QGES_DIR) not in sys.path:
        sys.path.insert(0, str(_QGES_DIR))
    from models.gpt import GPTQE, GPTConfig                        # type: ignore
    from training.train_gptqe import (                              # type: ignore
        get_subsequence_energies, circuit_resources,
    )
    return GPTQE, GPTConfig, get_subsequence_energies, circuit_resources


class GQERunner(BaseRunner):
    """Run Q-GESolver (GPTQE) on one molecule / seed."""

    def __init__(
        self,
        config_path: Path,
        mol_path: Path,
        result_dir: Path,
        seed: int,
        device=None,
    ):
        self.config_path = Path(config_path)

        cp = configparser.ConfigParser()
        cp.read(str(self.config_path))

        # ── Molecular / env config ────────────────────────────────────────────
        self.accept_err       = cp.getfloat("env", "accept_err",      fallback=0.0016)
        self.n_qubits         = cp.getint  ("env", "num_qubits")
        self.active_electrons = cp.getint  ("env", "active_electrons")
        self.active_orbitals  = cp.getint  ("env", "active_orbitals",
                                             fallback=self.n_qubits // 2)

        # ── Model hyperparameters ─────────────────────────────────────────────
        self.seq_len    = cp.getint("model", "seq_len",     fallback=6)
        self.train_size = cp.getint("model", "train_size",  fallback=2048)
        self.n_layer    = cp.getint("model", "n_layer",     fallback=6)
        self.n_head     = cp.getint("model", "n_head",      fallback=8)
        self.n_embd     = cp.getint("model", "n_embd",      fallback=256)

        # ── Training parameters ───────────────────────────────────────────────
        self.epochs           = cp.getint  ("general", "epochs",           fallback=10000)
        self.eval_every       = cp.getint  ("general", "eval_every",       fallback=100)
        self.eval_n_sequences = cp.getint  ("general", "eval_n_sequences", fallback=500)
        self.lr               = cp.getfloat("general", "lr",               fallback=3e-4)

        # PyTorch device for the GPT model.  PennyLane quantum simulation is
        # always CPU-based regardless of this setting.
        if device is not None and not isinstance(device, str):
            self.torch_device = str(device)
        else:
            self.torch_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Minimal conf dict required by BaseRunner.
        _conf = {
            "env":     {"accept_err": self.accept_err, "num_qubits": self.n_qubits},
            "general": {"compute_pcd": "0"},
        }

        self._gpt      = None   # set after training
        self._mol_data = None   # set during run()

        super().__init__(_conf, mol_path, result_dir, seed)

    # ── Abstract interface ────────────────────────────────────────────────────

    def run(self) -> dict:
        """Full GQE training + result serialisation."""
        from .mol_adapter import build_pennylane_objects
        GPTQE, GPTConfig, get_subsequence_energies, circuit_resources = _import_qges()

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # ── Build PennyLane objects ───────────────────────────────────────────
        print(f"[GQE] Building PennyLane objects from {self.mol_path.name} …",
              flush=True)
        mol = build_pennylane_objects(
            self.mol_path, self.active_electrons, self.active_orbitals
        )
        self._mol_data = mol
        op_pool      = mol["op_pool"]
        hf_state     = mol["hf_state"]
        hamiltonian  = mol["hamiltonian"]
        n_qubits     = mol["n_qubits"]
        exact_energy = mol["exact_energy"]
        op_pool_size = len(op_pool)

        print(f"[GQE] n_qubits={n_qubits}  op_pool_size={op_pool_size}  "
              f"seq_len={self.seq_len}  train_size={self.train_size}",
              flush=True)
        print(f"[GQE] E_exact={exact_energy:.6f} Ha  "
              f"accept_err={self.accept_err*1000:.1f} mHa",
              flush=True)

        # ── Sample training data ──────────────────────────────────────────────
        rng = np.random.default_rng(self.seed)
        train_op_inds = rng.integers(op_pool_size, size=(self.train_size, self.seq_len))
        train_tokens  = np.concatenate(
            [np.zeros((self.train_size, 1), dtype=int), train_op_inds + 1], axis=1
        )
        train_op_seq = [[op_pool[int(i)] for i in row] for row in train_op_inds]

        print(f"[GQE] Computing {self.train_size} training sequence energies …",
              flush=True)
        train_energies = get_subsequence_energies(
            train_op_seq, hamiltonian, hf_state, n_qubits, log_interval=500
        )   # shape (train_size, seq_len)

        tokens_t   = torch.from_numpy(train_tokens).to(self.torch_device)
        energies_t = torch.from_numpy(train_energies).float().to(self.torch_device)

        # ── Build model ───────────────────────────────────────────────────────
        gpt = GPTQE(GPTConfig(
            vocab_size=op_pool_size + 1,
            block_size=self.seq_len,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            dropout=0.1,
            bias=False,
        )).to(self.torch_device)

        optimizer = gpt.configure_optimizers(
            weight_decay=0.01,
            learning_rate=self.lr,
            betas=(0.9, 0.999),
            device_type=self.torch_device,
        )

        # ── Training loop ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        losses:             list[float] = []
        err_mha_history:    list[float] = []
        cnot_history:       list[int]   = []
        depth_history:      list[int]   = []
        eval_records:       list[dict]  = []   # episodes meeting accept_err
        best_error_ha = float("inf")

        for epoch in range(self.epochs):
            gpt.train()
            epoch_loss = 0.0
            for tok_b, en_b in zip(
                torch.tensor_split(tokens_t, 16),
                torch.tensor_split(energies_t, 16),
            ):
                optimizer.zero_grad()
                loss = gpt.calculate_loss(tok_b, en_b)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss)

            if (epoch + 1) % self.eval_every == 0:
                gpt.eval()
                with torch.no_grad():
                    gen_tokens, _ = gpt.generate(
                        n_sequences=self.eval_n_sequences,
                        max_new_tokens=self.seq_len,
                        temperature=0.01,
                        device=self.torch_device,
                    )
                gen_inds = np.clip(
                    (gen_tokens[:, 1:] - 1).cpu().numpy(), 0, op_pool_size - 1
                )
                gen_op_seq = [[op_pool[int(i)] for i in row] for row in gen_inds]

                true_Es = get_subsequence_energies(
                    gen_op_seq, hamiltonian, hf_state, n_qubits, log_interval=0
                )[:, -1]   # (eval_n_sequences,) — full-sequence energy

                best_idx  = int(np.argmin(true_Es))
                best_E    = float(true_Es[best_idx])
                error_ha  = abs(best_E - exact_energy)
                err_mha   = error_ha * 1000
                best_ops  = gen_op_seq[best_idx]
                cnot_cnt, depth = circuit_resources(best_ops, hf_state, n_qubits)

                err_mha_history.append(err_mha)
                cnot_history.append(cnot_cnt)
                depth_history.append(depth)

                print(f"  Epoch {epoch+1:5d} | Loss {epoch_loss:.4f} | "
                      f"err {err_mha:.2f} mHa | CNOT {cnot_cnt} | depth {depth}",
                      flush=True)

                if error_ha < best_error_ha:
                    best_error_ha = error_ha
                    torch.save(gpt.state_dict(), self.result_dir / "best_model.pt")

                # Record as analysis episode if it meets the threshold.
                if error_ha <= self.accept_err:
                    eval_records.append({
                        "eval_epoch": epoch + 1,
                        "error_ha":   error_ha,
                        "best_ops":   best_ops,
                        "cnot_cnt":   cnot_cnt,
                    })

        t_total = time.perf_counter() - t0
        self._gpt = gpt

        # ── Save arrays ───────────────────────────────────────────────────────
        np.save(self.result_dir / "losses.npy",                losses)
        np.save(self.result_dir / "best_energy_error_mHa.npy", err_mha_history)
        np.save(self.result_dir / "best_cnot_count.npy",       cnot_history)
        np.save(self.result_dir / "best_depth.npy",            depth_history)

        result = {
            "method":          "GQE",
            "seed":            self.seed,
            "best_energy_ha":  exact_energy + best_error_ha,
            "energy_error_ha": best_error_ha,
            "cnot_count":      cnot_history[-1] if cnot_history else -1,
            "circuit_depth":   depth_history[-1] if depth_history else -1,
            "nfev":            -1,
            "success":         int(best_error_ha < self.accept_err),
            "exact_energy_ha": exact_energy,
            "accept_err_ha":   self.accept_err,
        }

        self.save_result(result)
        self._write_run_meta(t_total, op_pool_size)
        self._write_config_used_cfg()
        self._write_episode_traces(eval_records)

        print(f"\n[GQE] Best error = {best_error_ha*1000:.4f} mHa  "
              f"({'✓ chem. accuracy' if result['success'] else '✗'})")
        print(f"[GQE] Saved → {self.result_dir}")
        return result

    def greedy_episode(self, env) -> dict:  # noqa: ARG002
        """Deterministic generation (very low temperature ≈ argmax)."""
        return self._generate_episode(temperature=1e-3, seed=None)

    def stochastic_episode(self, env, seed: int) -> dict:  # noqa: ARG002
        """Stochastic generation with a fixed seed (for PCD diversity)."""
        return self._generate_episode(temperature=1.0, seed=seed)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _generate_episode(self, temperature: float, seed: int | None) -> dict:
        if self._gpt is None:
            raise RuntimeError("Call run() before greedy/stochastic_episode().")
        _, _, get_subsequence_energies, circuit_resources = _import_qges()

        mol = self._mol_data
        op_pool      = mol["op_pool"]
        op_pool_size = len(op_pool)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self._gpt.eval()
        with torch.no_grad():
            gen_tokens, _ = self._gpt.generate(
                n_sequences=1,
                max_new_tokens=self.seq_len,
                temperature=temperature,
                device=self.torch_device,
            )
        op_inds = np.clip((gen_tokens[:, 1:] - 1).cpu().numpy(), 0, op_pool_size - 1)
        ops = [op_pool[int(i)] for i in op_inds[0]]

        true_Es = get_subsequence_energies(
            [ops], mol["hamiltonian"], mol["hf_state"], mol["n_qubits"], log_interval=0
        )
        energy   = float(true_Es[0, -1])
        error_ha = abs(energy - mol["exact_energy"])
        cnot_cnt, depth = circuit_resources(ops, mol["hf_state"], mol["n_qubits"])

        return {
            "energy":         energy,
            "energy_error":   error_ha,
            "cnot_count":     cnot_cnt,
            "rotation_count": 0,
            "success":        int(error_ha < self.accept_err),
            "steps":          self.seq_len,
            "op_history":     self._ops_to_gates_direct(ops),
            "n_qubits":       mol["n_qubits"],
            "final_state":    None,   # no PSQASBench state tensor for GQE
        }

    @staticmethod
    def _ops_to_gates_direct(ops) -> list[dict]:
        """Convert a PennyLane op list to gates_direct-compatible dicts.

        Gate type is "ucc_single" or "ucc_double".  The extra "step" key is
        used for ordering but is ignored by gate_dicts_to_gates() in the
        critical_structure_tool (unknown keys are skipped).
        """
        result = []
        for step, op in enumerate(ops):
            wires = [int(w) for w in op.wires]
            t     = float(op.parameters[0]) if op.parameters else 0.0
            gtype = "ucc_double" if "Double" in op.name else "ucc_single"
            result.append({"type": gtype, "wires": wires, "time": t, "step": step})
        return result

    # ── Result file writers ───────────────────────────────────────────────────

    def _write_run_meta(self, t_total: float, op_pool_size: int) -> None:
        path = self.result_dir / "run_meta.txt"
        with open(path, "w") as f:
            f.write(f"method                     = GQE\n")
            f.write(f"mol_path                   = {self.mol_path}\n")
            f.write(f"seed                       = {self.seed}\n")
            f.write(f"exact_energy_ha            = {self.exact_energy:.8f}\n")
            f.write(f"accept_err_ha              = {self.accept_err:.6f}\n")
            f.write(f"analysis_save_threshold_ha = {self.accept_err:.6f}\n")
            f.write(f"n_qubits                   = {self.n_qubits}\n")
            f.write(f"active_electrons           = {self.active_electrons}\n")
            f.write(f"active_orbitals            = {self.active_orbitals}\n")
            f.write(f"op_pool_size               = {op_pool_size}\n")
            f.write(f"seq_len                    = {self.seq_len}\n")
            f.write(f"train_size                 = {self.train_size}\n")
            f.write(f"n_layer                    = {self.n_layer}\n")
            f.write(f"n_head                     = {self.n_head}\n")
            f.write(f"n_embd                     = {self.n_embd}\n")
            f.write(f"epochs                     = {self.epochs}\n")
            f.write(f"lr                         = {self.lr}\n")
            f.write(f"total_time_s               = {t_total:.1f}\n")

    def _write_config_used_cfg(self) -> None:
        mol_file = Path(self.mol_path).name
        lines = [
            "[env]",
            f"num_qubits        = {self.n_qubits}",
            f"accept_err        = {self.accept_err}",
            f"active_electrons  = {self.active_electrons}",
            "connectivity      = all",
            "",
            "[problem]",
            f"mol_file = {mol_file}",
        ]
        (self.result_dir / "config_used.cfg").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def _write_episode_traces(self, eval_records: list[dict]) -> None:
        """Write episode_traces.txt compatible with critical_structure_tool.

        Each eval epoch that found a circuit below accept_err becomes one
        episode.  The analysis_snapshot stores the UCC operator sequence as
        ``gates_direct`` (type "ucc_single" / "ucc_double").

        The critical_structure_tool silently skips unknown gate types, so the
        standard circuit-level analysis (qubit heatmaps, CNOT edges) will
        produce empty results for GQE — this is expected.  Use the
        separate operator-pool analysis for GQE-specific insights.
        """
        lines: list[str] = []
        for ep_idx, rec in enumerate(eval_records):
            error_ha    = rec["error_ha"]
            gates_direct = self._ops_to_gates_direct(rec["best_ops"])
            snapshot    = {"step": rec["eval_epoch"], "gates_direct": gates_direct}
            lines.append(f"[episode {ep_idx}]")
            lines.append(f"energy_errors_ha = [{error_ha!r}]")
            lines.append(f"analysis_snapshots = [{snapshot!r}]")
            lines.append("")
        (self.result_dir / "episode_traces.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )
