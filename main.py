import sys
import torch

from bench_utils import (
    MOL_DIR, CONFIG_DIR, RESULT_DIR, MOL_FILES,
    METHOD_CONFIG_DIR, redirect_output, parse_args, get_runner,
)

def main():
    print("=" * 60)
    print("  PSQASBench — Pauli String QAS Benchmark")
    print("=" * 60)
    args = parse_args()

    # --- validate paths before redirecting (errors go to terminal) --------------
    device = torch.device(args['device'])

    mol_path = MOL_DIR / MOL_FILES[args['mol']]
    if not mol_path.exists():
        sys.exit(f"[ERROR] mol file not found: {mol_path}")

    config_subdir = METHOD_CONFIG_DIR.get(args['method'], args['method'])
    config_path = CONFIG_DIR / config_subdir / args['config']
    if not config_path.exists():
        sys.exit(f"[ERROR] config not found: {config_path}")

    # --- run with log file ---------------------------------------------------
    seed = args['seed']
    config_tag = config_path.stem
    result_dir = RESULT_DIR / args['method'] / args['mol'] / config_tag / f"seed{seed}"
    log_path   = result_dir / "run.log"
    log = redirect_output(log_path)
    print(f"[PSQASBench] method={args['method']}  mol={args['mol']}  "
          f"config={args['config']}  seed={seed}  device={device}")
    runner = get_runner(args['method'], config_path, mol_path,
                        result_dir, seed, device)
    result = runner.run()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log.close()
    return result

if __name__ == "__main__":
    main()
