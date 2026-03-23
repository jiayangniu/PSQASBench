import sys
import datetime

import numpy as np
import torch

from bench_utils import (
    MOL_DIR, CONFIG_DIR, RESULT_DIR, MOL_FILES,
    redirect_stdout, prompt_args, get_runner,
)


def main():
    print("=" * 60)
    print("  PSQASBench — Pauli String QAS Benchmark")
    print("=" * 60)
    args = prompt_args()

    # --- validate paths before redirecting (errors go to terminal) --------------
    device = torch.device(args['device'])

    mol_path = MOL_DIR / MOL_FILES[args['mol']]
    if not mol_path.exists():
        sys.exit(f"[ERROR] mol file not found: {mol_path}")

    config_path = CONFIG_DIR / args['method'] / args['config']
    if not config_path.exists():
        sys.exit(f"[ERROR] config not found: {config_path}")

    # --- per-seed runs, each with its own log file ----------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    for seed in args['seeds']:
        result_dir = RESULT_DIR / args['method'] / args['mol'] / f"seed{seed}"
        log_path   = result_dir / f"run_{timestamp}.log"
        log = redirect_stdout(log_path)

        print(f"[PSQASBench] method={args['method']}  mol={args['mol']}  "
              f"seed={seed}  device={device}")
        runner = get_runner(args['method'], config_path, mol_path,
                            result_dir, seed, device)
        results.append(runner.run())
        log.close()

    # --- summary log (saved alongside seed dirs) ------------------------------
    summary_path = RESULT_DIR / args['method'] / args['mol'] / f"summary_{timestamp}.log"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    errors    = [r["energy_error"] for r in results]
    successes = [r["success"]      for r in results]
    cnots     = [r["cnot_count"]   for r in results]
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"[SUMMARY] {args['method'].upper()} on {args['mol']}  ({len(args['seeds'])} seeds)\n")
        f.write(f"  seeds         : {args['seeds']}\n")
        f.write(f"  success rate  : {sum(successes)}/{len(successes)}\n")
        f.write(f"  energy error  : mean={np.mean(errors)*1000:.3f} mHa  "
                f"std={np.std(errors)*1000:.3f} mHa\n")
        f.write(f"  CNOT count    : mean={np.mean(cnots):.1f}  std={np.std(cnots):.1f}\n")
        f.write("=" * 60 + "\n")
    return results


if __name__ == "__main__":
    main()
