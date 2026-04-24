from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
import torch
from scipy import optimize

from RLQAS import VQE as vc


@dataclass
class PreparedCircuit:
    state_tensor: torch.Tensor
    warm_start: np.ndarray
    seed: int
    trainable_mask: np.ndarray | None = None


class DirectGateBatchOptimizer:
    def __init__(
        self,
        *,
        n_qubits: int,
        hamiltonian,
        exact_energy: float,
        energy_shift: float,
        device: torch.device,
        optim_alg: str,
        global_iters: int,
        n_restarts: int,
        rotosolve_sweeps: int,
        options: dict | None = None,
        global_batched_cobyla: bool = False,
        global_batched_rotosolve: bool = True,
        global_batched_spsa: bool = True,
        global_batched_psr: bool = True,
        parallel_eval_batch_size: int = 0,
    ):
        self.n_qubits = int(n_qubits)
        self.hamiltonian = hamiltonian
        self.exact_energy = float(exact_energy)
        self.energy_shift = float(energy_shift)
        self.device = device
        self.optim_alg = str(optim_alg).upper()
        self.global_iters = int(global_iters)
        self.n_restarts = int(n_restarts)
        self.rotosolve_sweeps = int(rotosolve_sweeps)
        self.options = dict(options or {})
        self.global_batched_cobyla = bool(global_batched_cobyla)
        self.global_batched_rotosolve = bool(global_batched_rotosolve)
        self.global_batched_spsa = bool(global_batched_spsa)
        self.global_batched_psr = bool(global_batched_psr)
        self.parallel_eval_batch_size = int(parallel_eval_batch_size)
        self._bvqe = vc.BatchedVQE(
            self.n_qubits,
            self.hamiltonian,
            self.energy_shift,
            self.device,
        )

    @staticmethod
    def auto_spsa_batch_size(n_params: int) -> int:
        n = max(1, int(n_params))
        if n <= 8:
            return n
        return max(1, min(32, int(np.ceil(n / 2.0))))

    def optimize_many(self, prepared: list[PreparedCircuit]) -> list[dict]:
        if not prepared:
            return []

        alg = self.optim_alg
        if alg == "COBYLA":
            return [self._optimize_one_cobyla(item) for item in prepared]

        if alg == "ROTOSOLVE":
            chunk_fn = (
                self._optimize_chunk_rotosolve
                if self.global_batched_rotosolve
                else lambda chunk: [self._optimize_chunk_rotosolve([item])[0] for item in chunk]
            )
            return self._optimize_in_chunks(prepared, chunk_fn)

        if alg in {"SPSA", "ADAMSPSA"}:
            chunk_fn = (
                lambda chunk: self._optimize_chunk_spsa(chunk, method=alg)
                if self.global_batched_spsa
                else [self._optimize_chunk_spsa([item], method=alg)[0] for item in chunk]
            )
            return self._optimize_in_chunks(prepared, chunk_fn)

        if alg == "PSRADAM":
            chunk_fn = (
                self._optimize_chunk_psr_adam
                if self.global_batched_psr
                else lambda chunk: [self._optimize_chunk_psr_adam([item])[0] for item in chunk]
            )
            return self._optimize_in_chunks(prepared, chunk_fn)

        raise ValueError(f"Unsupported direct optimizer '{self.optim_alg}'.")

    def _optimize_in_chunks(self, prepared: list[PreparedCircuit], chunk_fn) -> list[dict]:
        chunk_size = self._chunk_size(len(prepared))
        results: list[dict] = []
        for start in range(0, len(prepared), chunk_size):
            results.extend(chunk_fn(prepared[start : start + chunk_size]))
        return results

    def _chunk_size(self, total: int) -> int:
        if self.parallel_eval_batch_size > 0:
            return max(1, min(total, self.parallel_eval_batch_size))
        if self.device.type == "cuda" and self.optim_alg in {"ROTOSOLVE", "SPSA", "ADAMSPSA", "PSRADAM"}:
            return max(1, min(total, 8))
        return 1

    def _use_streams(self, n_items: int) -> bool:
        return torch.cuda.is_available() and self.device.type == "cuda" and n_items > 1

    def _energy_batch(self, state_gpu: torch.Tensor, angle_matrix: np.ndarray) -> np.ndarray:
        probe_t = torch.tensor(angle_matrix, dtype=torch.float32, device=self.device)
        return self._bvqe.eval_batch(state_gpu, probe_t).detach().cpu().numpy()

    def _energy_single(self, state_gpu: torch.Tensor, angles: np.ndarray) -> float:
        probe = np.asarray(angles, dtype=np.float32).reshape(1, -1)
        return float(self._energy_batch(state_gpu, probe)[0])

    def _empty_result(self, item: PreparedCircuit) -> dict:
        state_gpu = item.state_tensor.to(self.device)
        total_energy = self._energy_single(state_gpu, np.empty((0,), dtype=np.float32))
        return {
            "energy": total_energy,
            "energy_error": abs(total_energy - self.exact_energy),
            "nfev": 1,
            "params": np.array([], dtype=np.float64),
        }

    def _optimize_one_cobyla(self, item: PreparedCircuit) -> dict:
        warm_start = np.asarray(item.warm_start, dtype=np.float32)
        if warm_start.size == 0:
            return self._empty_result(item)

        state_gpu = item.state_tensor.to(self.device)
        rng = np.random.default_rng(item.seed)
        initial_points = [warm_start.astype(np.float64)]
        for _ in range(max(1, self.n_restarts) - 1):
            initial_points.append(rng.uniform(-np.pi, np.pi, warm_start.size).astype(np.float64))

        best_fun = float("inf")
        best_params = initial_points[0].copy()
        total_nfev = 0

        for x0 in initial_points:
            nfev_count = [0]

            def cost(params):
                nfev_count[0] += 1
                return self._energy_single(state_gpu, np.asarray(params, dtype=np.float32))

            res = optimize.minimize(
                cost,
                x0,
                method="COBYLA",
                options={"maxiter": int(self.global_iters), "rhobeg": 1.0},
            )
            total_nfev += int(nfev_count[0])
            fun = float(res.fun) if np.isfinite(res.fun) else float(cost(x0))
            params = np.asarray(res.x if res.x is not None else x0, dtype=np.float64)
            if fun < best_fun:
                best_fun = fun
                best_params = params

        return {
            "energy": float(best_fun),
            "energy_error": abs(float(best_fun) - self.exact_energy),
            "nfev": int(total_nfev),
            "params": best_params,
        }

    def _optimize_chunk_rotosolve(self, chunk: list[PreparedCircuit]) -> list[dict]:
        results: list[dict | None] = [None] * len(chunk)
        meta = []
        for idx, item in enumerate(chunk):
            warm_start = np.asarray(item.warm_start, dtype=np.float32)
            if warm_start.size == 0:
                results[idx] = self._empty_result(item)
                continue
            meta.append(
                {
                    "idx": idx,
                    "state_gpu": item.state_tensor.to(self.device),
                    "angles": warm_start.copy(),
                    "n": int(warm_start.size),
                    "nfev": 0,
                }
            )

        if meta:
            use_streams = self._use_streams(len(meta))
            streams = [torch.cuda.Stream() for _ in meta] if use_streams else [None] * len(meta)
            probe_vals = np.array([0.0, np.pi / 2.0, np.pi], dtype=np.float32)

            for _ in range(max(1, self.rotosolve_sweeps)):
                e_tensors = [None] * len(meta)
                for i, (m, stream) in enumerate(zip(meta, streams)):
                    probe = np.tile(m["angles"], (3 * m["n"], 1))
                    for param_idx in range(m["n"]):
                        probe[3 * param_idx : 3 * param_idx + 3, param_idx] = probe_vals
                    ctx = torch.cuda.stream(stream) if stream else contextlib.nullcontext()
                    with ctx:
                        probe_t = torch.tensor(probe, dtype=torch.float32, device=self.device)
                        e_tensors[i] = self._bvqe.eval_batch(m["state_gpu"], probe_t)
                if use_streams:
                    torch.cuda.synchronize()

                for i, m in enumerate(meta):
                    energies = e_tensors[i].detach().cpu().numpy()
                    m["nfev"] += 3 * m["n"]
                    for param_idx in range(m["n"]):
                        e0 = float(energies[3 * param_idx])
                        epi2 = float(energies[3 * param_idx + 1])
                        epi = float(energies[3 * param_idx + 2])
                        A = (e0 - epi) / 2.0
                        C = (e0 + epi) / 2.0
                        B = epi2 - C
                        m["angles"][param_idx] = float(np.arctan2(-B, -A))

            final_tensors = [None] * len(meta)
            for i, (m, stream) in enumerate(zip(meta, streams)):
                ctx = torch.cuda.stream(stream) if stream else contextlib.nullcontext()
                with ctx:
                    final_t = torch.tensor(m["angles"].reshape(1, -1), dtype=torch.float32, device=self.device)
                    final_tensors[i] = self._bvqe.eval_batch(m["state_gpu"], final_t)
            if use_streams:
                torch.cuda.synchronize()

            for i, m in enumerate(meta):
                total_energy = float(final_tensors[i].item())
                m["nfev"] += 1
                results[m["idx"]] = {
                    "energy": total_energy,
                    "energy_error": abs(total_energy - self.exact_energy),
                    "nfev": int(m["nfev"]),
                    "params": m["angles"].astype(np.float64, copy=False),
                }

        return [result for result in results if result is not None]

    def _optimize_chunk_psr_adam(self, chunk: list[PreparedCircuit]) -> list[dict]:
        results: list[dict | None] = [None] * len(chunk)
        meta = []
        for idx, item in enumerate(chunk):
            warm_start = np.asarray(item.warm_start, dtype=np.float32)
            if warm_start.size == 0:
                results[idx] = self._empty_result(item)
                continue
            rng = np.random.default_rng(item.seed)
            zero_mask = np.abs(warm_start) < 1e-6
            if zero_mask.any():
                warm_start = warm_start.copy()
                warm_start[zero_mask] = rng.uniform(-0.3, 0.3, int(zero_mask.sum())).astype(np.float32)
            meta.append(
                {
                    "idx": idx,
                    "state_gpu": item.state_tensor.to(self.device),
                    "angles": warm_start.reshape(1, -1).copy(),
                    "n": int(warm_start.size),
                    "lr": float(self.options.get("lr", 0.01)),
                    "b1": float(self.options.get("beta_1", 0.9)),
                    "b2": float(self.options.get("beta_2", 0.999)),
                    "steps": max(1, int(self.global_iters)),
                    "m": np.zeros((1, warm_start.size), dtype=np.float32),
                    "v": np.zeros((1, warm_start.size), dtype=np.float32),
                    "best_val": np.inf,
                    "best_angles": warm_start.reshape(1, -1).copy(),
                    "nfev": 0,
                }
            )

        if meta:
            use_streams = self._use_streams(len(meta))
            streams = [torch.cuda.Stream() for _ in meta] if use_streams else [None] * len(meta)

            init_tensors = [None] * len(meta)
            for i, (m, stream) in enumerate(zip(meta, streams)):
                ctx = torch.cuda.stream(stream) if stream else contextlib.nullcontext()
                with ctx:
                    a_t = torch.tensor(m["angles"], dtype=torch.float32, device=self.device)
                    init_tensors[i] = self._bvqe.eval_batch(m["state_gpu"], a_t)
            if use_streams:
                torch.cuda.synchronize()
            for i, m in enumerate(meta):
                m["best_val"] = float(init_tensors[i].item())
                m["nfev"] += 1

            max_steps = max(m["steps"] for m in meta)
            for step in range(max_steps):
                e_tensors = [None] * len(meta)
                for i, (m, stream) in enumerate(zip(meta, streams)):
                    if step >= m["steps"]:
                        continue
                    # Row 0: current angles (for best-val tracking).
                    # Rows 1,3,…,2n-1: θ_j + π/2  (PSR forward shift).
                    # Rows 2,4,…,2n  : θ_j - π/2  (PSR backward shift).
                    psr_probe = np.repeat(m["angles"], 2 * m["n"], axis=0)
                    for param_idx in range(m["n"]):
                        psr_probe[2 * param_idx, param_idx] += np.pi / 2
                        psr_probe[2 * param_idx + 1, param_idx] -= np.pi / 2
                    probe = np.concatenate([m["angles"], psr_probe], axis=0)
                    ctx = torch.cuda.stream(stream) if stream else contextlib.nullcontext()
                    with ctx:
                        probe_t = torch.tensor(probe, dtype=torch.float32, device=self.device)
                        e_tensors[i] = self._bvqe.eval_batch(m["state_gpu"], probe_t)
                if use_streams:
                    torch.cuda.synchronize()

                for i, m in enumerate(meta):
                    if e_tensors[i] is None:
                        continue
                    energies = e_tensors[i].detach().cpu().numpy()
                    m["nfev"] += 2 * m["n"] + 1
                    val = float(energies[0])
                    if val < m["best_val"]:
                        m["best_val"] = val
                        m["best_angles"] = m["angles"].copy()
                    # energies[1::2] → +π/2 evals, energies[2::2] → -π/2 evals
                    grad = ((energies[1::2] - energies[2::2]) / 2.0).reshape(1, -1)
                    m["m"] = m["b1"] * m["m"] + (1.0 - m["b1"]) * grad
                    m["v"] = m["b2"] * m["v"] + (1.0 - m["b2"]) * grad ** 2
                    mhat = m["m"] / (1.0 - m["b1"] ** (step + 1))
                    vhat = m["v"] / (1.0 - m["b2"] ** (step + 1))
                    m["angles"] = m["angles"] - m["lr"] * mhat / (np.sqrt(vhat) + 1e-8)

            final_tensors = [None] * len(meta)
            for i, (m, stream) in enumerate(zip(meta, streams)):
                ctx = torch.cuda.stream(stream) if stream else contextlib.nullcontext()
                with ctx:
                    a_t = torch.tensor(m["angles"], dtype=torch.float32, device=self.device)
                    final_tensors[i] = self._bvqe.eval_batch(m["state_gpu"], a_t)
            if use_streams:
                torch.cuda.synchronize()

            for i, m in enumerate(meta):
                final_val = float(final_tensors[i].item())
                m["nfev"] += 1
                if final_val < m["best_val"]:
                    m["best_val"] = final_val
                    m["best_angles"] = m["angles"].copy()
                results[m["idx"]] = {
                    "energy": float(m["best_val"]),
                    "energy_error": abs(float(m["best_val"]) - self.exact_energy),
                    "nfev": int(m["nfev"]),
                    "params": m["best_angles"].flatten().astype(np.float64, copy=False),
                }

        return [result for result in results if result is not None]

    def _optimize_chunk_spsa(self, chunk: list[PreparedCircuit], *, method: str) -> list[dict]:
        results: list[dict | None] = [None] * len(chunk)
        use_adam = str(method).upper() == "ADAMSPSA"
        meta = []
        for idx, item in enumerate(chunk):
            warm_start = np.asarray(item.warm_start, dtype=np.float32)
            if warm_start.size == 0:
                results[idx] = self._empty_result(item)
                continue
            n_params = int(warm_start.size)
            meta.append(
                {
                    "idx": idx,
                    "state_gpu": item.state_tensor.to(self.device),
                    "angles": warm_start.reshape(1, -1).copy(),
                    "rng": np.random.default_rng(item.seed),
                    "a": float(self.options.get("a", 0.05)),
                    "alpha": float(self.options.get("alpha", 0.602)),
                    "c": float(self.options.get("c", 0.1)),
                    "gamma": float(self.options.get("gamma", 0.101)),
                    "A": float(self.options.get("lamda", max(10.0, self.global_iters * 0.1))),
                    "b1": float(self.options.get("beta_1", 0.9)),
                    "b2": float(self.options.get("beta_2", 0.999)),
                    "spsa_batch": int(self.auto_spsa_batch_size(n_params)),
                    "iters": max(1, int(self.global_iters)),
                    "m": np.zeros((1, n_params), dtype=np.float32),
                    "v": np.zeros((1, n_params), dtype=np.float32),
                    "best_val": np.inf,
                    "best_angles": warm_start.reshape(1, -1).copy(),
                    "nfev": 0,
                }
            )

        if meta:
            use_streams = self._use_streams(len(meta))
            streams = [torch.cuda.Stream() for _ in meta] if use_streams else [None] * len(meta)

            init_tensors = [None] * len(meta)
            for i, (m, stream) in enumerate(zip(meta, streams)):
                ctx = torch.cuda.stream(stream) if stream else contextlib.nullcontext()
                with ctx:
                    init_tensors[i] = self._bvqe.eval_batch(
                        m["state_gpu"],
                        torch.tensor(m["angles"], dtype=torch.float32, device=self.device),
                    )
            if use_streams:
                torch.cuda.synchronize()
            for i, m in enumerate(meta):
                m["best_val"] = float(init_tensors[i].item())
                m["nfev"] += 1

            max_iters = max(m["iters"] for m in meta)
            for epoch in range(max_iters):
                e_merged = [None] * len(meta)
                for i, (m, stream) in enumerate(zip(meta, streams)):
                    if epoch >= m["iters"]:
                        continue
                    ck = vc._spsa_ck(epoch, c=m["c"], gamma=m["gamma"])
                    batch_size = m["spsa_batch"]
                    delta = m["rng"].choice(
                        [-1.0, 1.0],
                        size=(batch_size, m["angles"].shape[1]),
                    ).astype(np.float32)
                    m["_delta"] = delta
                    m["_ck"] = ck
                    probe = np.concatenate(
                        [
                            m["angles"] + ck * delta,
                            m["angles"] - ck * delta,
                            m["angles"],
                        ],
                        axis=0,
                    )
                    ctx = torch.cuda.stream(stream) if stream else contextlib.nullcontext()
                    with ctx:
                        probe_t = torch.tensor(probe, dtype=torch.float32, device=self.device)
                        e_merged[i] = self._bvqe.eval_batch(m["state_gpu"], probe_t)
                if use_streams:
                    torch.cuda.synchronize()

                for i, m in enumerate(meta):
                    if epoch >= m["iters"] or e_merged[i] is None:
                        continue
                    e_all = e_merged[i].detach().cpu().numpy()
                    batch_size = m["spsa_batch"]
                    val = float(e_all[2 * batch_size])
                    m["nfev"] += 2 * batch_size + 1
                    if val < m["best_val"]:
                        m["best_val"] = val
                        m["best_angles"] = m["angles"].copy()

                    ak = vc._spsa_lr(epoch, a=m["a"], A=m["A"], alpha=m["alpha"])
                    ck = m["_ck"]
                    delta = m["_delta"]
                    e_plus = e_all[:batch_size].reshape(-1, 1)
                    e_minus = e_all[batch_size : 2 * batch_size].reshape(-1, 1)
                    grad_dirs = ((e_plus - e_minus) / (2.0 * ck)) / delta
                    grad = grad_dirs.mean(axis=0, keepdims=True)
                    if use_adam:
                        m["m"] = m["b1"] * m["m"] + (1.0 - m["b1"]) * grad
                        m["v"] = m["b2"] * m["v"] + (1.0 - m["b2"]) * grad ** 2
                        mhat = m["m"] / (1.0 - m["b1"] ** (epoch + 1))
                        vhat = m["v"] / (1.0 - m["b2"] ** (epoch + 1))
                        m["angles"] = m["angles"] - ak * mhat / (np.sqrt(vhat) + 1e-8)
                    else:
                        m["angles"] = m["angles"] - ak * grad

            final_tensors = [None] * len(meta)
            for i, (m, stream) in enumerate(zip(meta, streams)):
                ctx = torch.cuda.stream(stream) if stream else contextlib.nullcontext()
                with ctx:
                    final_tensors[i] = self._bvqe.eval_batch(
                        m["state_gpu"],
                        torch.tensor(m["angles"], dtype=torch.float32, device=self.device),
                    )
            if use_streams:
                torch.cuda.synchronize()

            for i, m in enumerate(meta):
                final_val = float(final_tensors[i].item())
                m["nfev"] += 1
                if final_val < m["best_val"]:
                    m["best_val"] = final_val
                    m["best_angles"] = m["angles"].copy()
                results[m["idx"]] = {
                    "energy": float(m["best_val"]),
                    "energy_error": abs(float(m["best_val"]) - self.exact_energy),
                    "nfev": int(m["nfev"]),
                    "params": m["best_angles"].flatten().astype(np.float64, copy=False),
                }

        return [result for result in results if result is not None]
