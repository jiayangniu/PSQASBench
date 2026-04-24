"""Structured plain-text result writers for PSQASBench runs."""

from __future__ import annotations

import copy
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _fmt_scalar(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _dump_json(value) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False)


class ResultLogger:
    """Write agreed text/TSV result artifacts into one result directory."""

    EPISODE_SUMMARY_FIELDS = [
        "episode",
        "final_energy_ha",
        "final_energy_error_ha",
        "episode_return",
        "depth",
        "cnot_count",
        "rotation_count",
        "n_steps",
        "epsilon_end",
        "episode_time_sec",
        "is_global_best",
    ]

    POLICY_LOSS_FIELDS = ["update_idx", "episode", "step", "policy_loss"]

    EVAL_TREND_FIELDS = [
        "eval_episode",
        "best_error_mha",
        "D_struct",
        "D_func",
        "sr_at_chem",
        "cnot_at_chem",
    ]

    def __init__(
        self,
        result_dir: Path,
        config_path: Path,
        *,
        method: str,
        mol: str,
        seed: int,
        device: str,
        exact_energy: float,
        accept_err: float,
        analysis_save_threshold: float | None,
        save_summary_detailed: bool,
        runtime_overrides: dict | None = None,
    ):
        self.result_dir = Path(result_dir)
        self.config_path = Path(config_path)
        self.method = method
        self.mol = mol
        self.seed = int(seed)
        self.device = str(device)
        self.exact_energy = float(exact_energy)
        self.accept_err = float(accept_err)
        self.analysis_save_threshold = (
            None if analysis_save_threshold is None else float(analysis_save_threshold)
        )
        self.save_summary_detailed = bool(save_summary_detailed)
        self.runtime_overrides = runtime_overrides or {}

        self.run_start_iso = _now_utc_iso()
        self.wandb_run_name = ""
        self.wandb_run_id = ""
        self._policy_update_idx = 0
        self._best_train = None
        self._best_eval = None
        self._eval_trend = []

        self.run_meta_path = self.result_dir / "run_meta.txt"
        self.episode_summary_path = self.result_dir / "episode_summary.tsv"
        self.episode_traces_path = self.result_dir / "episode_traces.txt"
        self.policy_loss_path = self.result_dir / "policy_loss.tsv"
        self.best_train_path = self.result_dir / "best_train.txt"
        self.best_eval_path = self.result_dir / "best_eval.txt"

        self.result_dir.mkdir(parents=True, exist_ok=True)
        self._acquire_lock()
        self._copy_config()
        self._init_text_files()
        self.write_run_meta(status="running")

    def _acquire_lock(self) -> None:
        lock_path = self.result_dir / "run.lock"
        if lock_path.exists():
            try:
                existing_pid = int(lock_path.read_text(encoding="utf-8").strip())
                os.kill(existing_pid, 0)  # raises OSError if process is dead
                raise RuntimeError(
                    f"Result directory is already in use by PID {existing_pid}: {self.result_dir}\n"
                    f"If that process is no longer running, delete {lock_path} and retry."
                )
            except (ValueError, OSError):
                # Stale lock — process is dead, safe to overwrite
                pass
        lock_path.write_text(str(os.getpid()), encoding="utf-8")

    def release_lock(self) -> None:
        lock_path = self.result_dir / "run.lock"
        lock_path.unlink(missing_ok=True)

    def _copy_config(self):
        shutil.copyfile(self.config_path, self.result_dir / "config_used.cfg")

    def _init_text_files(self):
        # Explicitly unlink before write to guarantee a clean file even when
        # re-running into an existing result directory (e.g. after a crash).
        for path in [
            self.episode_traces_path,
            self.episode_summary_path,
            self.policy_loss_path,
            self.best_train_path,
            self.best_eval_path,
        ]:
            path.unlink(missing_ok=True)
        self.episode_summary_path.write_text(
            "\t".join(self.EPISODE_SUMMARY_FIELDS) + "\n",
            encoding="utf-8",
        )
        self.policy_loss_path.write_text(
            "\t".join(self.POLICY_LOSS_FIELDS) + "\n",
            encoding="utf-8",
        )
        self.episode_traces_path.write_text("", encoding="utf-8")
        self.best_train_path.write_text("", encoding="utf-8")
        self.best_eval_path.write_text("", encoding="utf-8")

    def set_wandb_run(self, run_name: str | None, run_id: str | None = None):
        self.wandb_run_name = run_name or ""
        self.wandb_run_id = run_id or ""
        self.write_run_meta(status="running")

    def write_run_meta(self, status: str, final_result: dict | None = None):
        lines = [
            f"status = {status}",
            f"method = {self.method}",
            f"mol = {self.mol}",
            f"seed = {self.seed}",
            f"device = {self.device}",
            f"result_dir = {self.result_dir}",
            f"config_name = {self.config_path.name}",
            f"start_time_utc = {self.run_start_iso}",
            f"exact_energy_ha = {self.exact_energy}",
            f"accept_err_ha = {self.accept_err}",
            f"analysis_save_threshold_ha = {_fmt_scalar(self.analysis_save_threshold)}",
            f"save_summary_detailed = {int(self.save_summary_detailed)}",
            f"wandb_run_name = {self.wandb_run_name}",
            f"wandb_run_id = {self.wandb_run_id}",
        ]
        if self.runtime_overrides:
            lines.append(f"runtime_overrides = {_dump_json(self.runtime_overrides)}")
        if final_result:
            for key, value in final_result.items():
                lines.append(f"{key} = {_fmt_scalar(value)}")
        self.run_meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def finalize_run_meta(self, wall_clock_sec: float, final_result: dict | None = None):
        final_fields = {
            "end_time_utc": _now_utc_iso(),
            "wall_clock_sec": wall_clock_sec,
        }
        if final_result:
            final_fields.update(final_result)
        self.write_run_meta(status="completed", final_result=final_fields)
        self.release_lock()

    def append_episode_summary(self, record: dict):
        row = [_fmt_scalar(record.get(field)) for field in self.EPISODE_SUMMARY_FIELDS]
        with self.episode_summary_path.open("a", encoding="utf-8") as f:
            f.write("\t".join(row) + "\n")

    def append_episode_trace(self, episode: int, trace: dict):
        payload = {
            "actions": trace.get("actions", []),
            "energies_ha": trace.get("energies", []),
            "energy_errors_ha": trace.get("energy_errors", []),
            "rewards": trace.get("rewards", []),
            "analysis_snapshots": trace.get("analysis_snapshots", []),
        }
        with self.episode_traces_path.open("a", encoding="utf-8") as f:
            f.write(f"[episode {int(episode)}]\n")
            for key, value in payload.items():
                f.write(f"{key} = {_dump_json(value)}\n")
            f.write("\n")

    def append_policy_loss(self, *, episode: int, step: int, policy_loss: float):
        self._policy_update_idx += 1
        record = {
            "update_idx": self._policy_update_idx,
            "episode": episode,
            "step": step,
            "policy_loss": policy_loss,
        }
        row = [_fmt_scalar(record.get(field)) for field in self.POLICY_LOSS_FIELDS]
        with self.policy_loss_path.open("a", encoding="utf-8") as f:
            f.write("\t".join(row) + "\n")

    def update_best_train(self, record: dict):
        self._best_train = copy.deepcopy(_json_safe(record))
        lines = [
            "[best_train]",
            f"episode = {_fmt_scalar(self._best_train.get('episode'))}",
            f"step = {_fmt_scalar(self._best_train.get('step'))}",
            f"energy_ha = {_fmt_scalar(self._best_train.get('energy'))}",
            f"energy_error_ha = {_fmt_scalar(self._best_train.get('energy_error'))}",
            f"depth = {_fmt_scalar(self._best_train.get('depth'))}",
            f"cnot_count = {_fmt_scalar(self._best_train.get('cnot_count'))}",
            f"rotation_count = {_fmt_scalar(self._best_train.get('rotation_count'))}",
            f"op_history = {_dump_json(self._best_train.get('op_history', []))}",
            "",
        ]
        self.best_train_path.write_text("\n".join(lines), encoding="utf-8")

    def update_eval(self, eval_metrics: dict):
        trend_row = {
            "eval_episode": eval_metrics["eval_episode"],
            "best_error_mha": eval_metrics["best_error_mha"],
            "D_struct": eval_metrics["D_struct"],
            "D_func": eval_metrics["D_func"],
            "sr_at_chem": eval_metrics["sr_at_chem"],
            "cnot_at_chem": eval_metrics["cnot_at_chem"],
        }
        self._eval_trend.append(copy.deepcopy(_json_safe(trend_row)))

        if (
            self._best_eval is None
            or float(eval_metrics["best_error_mha"]) < float(self._best_eval["best_error_mha"])
        ):
            self._best_eval = copy.deepcopy(_json_safe(eval_metrics))

        self._write_best_eval()

    def _write_best_eval(self):
        if self._best_eval is None:
            return

        lines = [
            "[best_eval_metrics]",
            f"eval_episode = {_fmt_scalar(self._best_eval.get('eval_episode'))}",
            f"sr_at_chem = {_fmt_scalar(self._best_eval.get('sr_at_chem'))}",
            f"cnot_at_chem = {_fmt_scalar(self._best_eval.get('cnot_at_chem'))}",
            f"best_error_mha = {_fmt_scalar(self._best_eval.get('best_error_mha'))}",
            f"mean_error_mha = {_fmt_scalar(self._best_eval.get('mean_error_mha'))}",
            f"mean_cnots = {_fmt_scalar(self._best_eval.get('mean_cnots'))}",
            f"D_struct = {_fmt_scalar(self._best_eval.get('D_struct'))}",
            f"D_func = {_fmt_scalar(self._best_eval.get('D_func'))}",
            "",
            "[best_eval_circuit]",
            f"energy_ha = {_fmt_scalar(self._best_eval.get('best_rollout_energy'))}",
            f"energy_error_ha = {_fmt_scalar(self._best_eval.get('best_rollout_energy_error'))}",
            f"depth = {_fmt_scalar(self._best_eval.get('best_rollout_depth'))}",
            f"cnot_count = {_fmt_scalar(self._best_eval.get('best_rollout_cnot_count'))}",
            f"rotation_count = {_fmt_scalar(self._best_eval.get('best_rollout_rotation_count'))}",
            f"op_history = {_dump_json(self._best_eval.get('best_rollout_op_history', []))}",
            "",
            "[eval_trend]",
            "\t".join(self.EVAL_TREND_FIELDS),
        ]

        for row in self._eval_trend:
            lines.append("\t".join(_fmt_scalar(row.get(field)) for field in self.EVAL_TREND_FIELDS))

        self.best_eval_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
