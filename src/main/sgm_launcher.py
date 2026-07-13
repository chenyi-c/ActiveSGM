#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

import mmengine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified launcher for ActiveSGM.")
    parser.add_argument("--dataset", type=str, default="Replica", choices=["Replica", "MP3D"])
    parser.add_argument("--scene", type=str, default="office0")
    parser.add_argument("--exp", type=str, default="ActiveSem")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable_vis", type=int, default=0, choices=[0, 1])
    parser.add_argument("--gpus", type=str, default="0", help="CUDA_VISIBLE_DEVICES value, e.g. 0 or 0,1")
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true", help="Only validate environment and config, do not launch training.")
    return parser.parse_args()


def _check_runtime_dependency(module_name: str) -> None:
    __import__(module_name)


def main() -> int:
    args = parse_args()
    proj_dir = Path(__file__).resolve().parents[2]
    cfg_path = proj_dir / "configs" / args.dataset / args.scene / f"{args.exp}.py"

    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}")
        return 1

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Minimal runtime checks for the current ActiveSGM route.
    try:
        _check_runtime_dependency("torch")
        _check_runtime_dependency("habitat_sim")
        _check_runtime_dependency("diff_gaussian_rasterization")
    except Exception as exc:
        print(f"[ERROR] Runtime dependency check failed: {exc}")
        return 1

    try:
        cfg = mmengine.Config.fromfile(str(cfg_path))
        print(f"[OK] Loaded config: {cfg_path}")
        print(f"[INFO] dataset={cfg.general.dataset}, scene={cfg.general.scene}, num_iter={cfg.general.num_iter}")
    except Exception as exc:
        print(f"[ERROR] Failed to parse config: {exc}")
        return 1

    if args.dry_run:
        print("[OK] Dry-run finished. Framework is ready.")
        return 0

    if args.result_dir is None:
        args.result_dir = str(proj_dir / "results" / args.dataset / args.scene / args.exp / "run_0")
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(proj_dir / "src/main/activesgm.py"),
        "--cfg",
        str(cfg_path),
        "--seed",
        str(args.seed),
        "--result_dir",
        args.result_dir,
        "--enable_vis",
        str(args.enable_vis),
    ]

    print(f"[INFO] Launch command: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(proj_dir), env=os.environ.copy())


if __name__ == "__main__":
    raise SystemExit(main())
