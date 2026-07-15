#!/usr/bin/env python3
"""Run plan for the exp5 IMPES comparison.

This script prepares the six requested tracks but does not run anything unless
you pass an explicit stage. Intended workflow:

    python run_exp5_impes.py --stage plan
    python run_exp5_impes.py --stage smoke
    python run_exp5_impes.py --stage run

The long runs are not launched by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SIM = ROOT / "impes_spe10_simulator.py"
PYTHON = Path("/home/muchamad/PhD/fenicsx/bin/python")
EXP_ROOT = ROOT / "impes_runs"
PREFLIGHT_ROOT = EXP_ROOT / "exp5_preflight"

SNAP_TIMES = [i / 100.0 for i in range(1, 11)]
M = 1.0
TRANSPORT_DT = 1.0e-5


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    method: str
    n_time: int
    dt_outer: float
    out_dir: Path
    pinn_mode: str | None = None
    viz_every: int = 1000
    save_every: int = 1000
    print_every: int = 1000

    @property
    def final_time(self) -> float:
        return float(self.n_time * self.dt_outer)

    @property
    def snapshot_steps(self) -> dict[float, int]:
        steps: dict[float, int] = {}
        for t in SNAP_TIMES:
            step_float = t / self.dt_outer
            step = int(round(step_float))
            if abs(step - step_float) > 1.0e-10:
                raise ValueError(f"{self.label}: snapshot time {t} is not on the outer grid")
            steps[t] = step
        return steps


RUNS = [
    RunSpec("cg", "CG", "CG", 10000, 1.0e-5, EXP_ROOT / "exp5_CG_M1"),
    RunSpec("frozen", "Frozen", "PINN", 10000, 1.0e-5, EXP_ROOT / "exp5_frozen_M1", pinn_mode="frozen"),
    RunSpec("nlr", "NLR@1", "NLR", 10000, 1.0e-5, EXP_ROOT / "exp5_NLR_M1"),
    RunSpec("linear", "PINN linear", "PINN", 10000, 1.0e-5, EXP_ROOT / "exp5_linear_M1", pinn_mode="last_layer"),
    RunSpec("pou", "PINN PoU@1", "PINN", 10000, 1.0e-5, EXP_ROOT / "exp5_pou_M1", pinn_mode="pou"),
    RunSpec("full", "PINN full @1000", "PINN", 10, 1.0e-2, EXP_ROOT / "exp5_full_M1", pinn_mode="full", viz_every=1, save_every=1, print_every=1),
    RunSpec("nlr_coarse", "NLR@1000", "NLR", 10, 1.0e-2, EXP_ROOT / "exp5_NLR_coarse_M1", viz_every=1, save_every=1, print_every=1),
]


def python_exe() -> str:
    return str(PYTHON if PYTHON.exists() else Path(sys.executable))


def base_cmd(
    spec: RunSpec,
    *,
    n_time: int | None = None,
    out_dir: Path | None = None,
    viz_every: int | None = None,
    save_every: int | None = None,
    print_every: int | None = None,
) -> list[str]:
    cmd = [
        python_exe(),
        str(SIM),
        "--method",
        spec.method,
        "--M",
        f"{M:g}",
        "--N_time",
        str(spec.n_time if n_time is None else n_time),
        "--DT_outer",
        f"{spec.dt_outer:.16g}",
        "--transport_dt",
        f"{TRANSPORT_DT:.16g}",
        "--viz_every",
        str(spec.viz_every if viz_every is None else viz_every),
        "--save_every",
        str(spec.save_every if save_every is None else save_every),
        "--print_every",
        str(spec.print_every if print_every is None else print_every),
        "--out_dir",
        str(spec.out_dir if out_dir is None else out_dir),
    ]
    if spec.pinn_mode is not None:
        cmd.extend(["--pinn_mode", spec.pinn_mode])
    return cmd


def command_text(cmd: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in cmd)


def assert_snapshot_cadence(spec: RunSpec) -> None:
    if abs(spec.final_time - 0.1) > 1.0e-12:
        raise AssertionError(f"{spec.label}: final time is {spec.final_time}, expected 0.1")
    for t, step in spec.snapshot_steps.items():
        if step < 1 or step > spec.n_time:
            raise AssertionError(f"{spec.label}: snapshot {t} maps to invalid step {step}")
        if step % spec.save_every != 0:
            raise AssertionError(
                f"{spec.label}: snapshot {t} at step {step} is not saved by save_every={spec.save_every}"
            )


def read_last_log_row(run_dir: Path) -> dict[str, str]:
    path = run_dir / "step_log.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows[-1]


def remove_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"{path} exists; pass --force to remove it before rerunning.")
        shutil.rmtree(path)


def run_cmd(cmd: list[str]) -> None:
    print(command_text(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    subprocess.run(cmd, check=True, env=env)


def stage_plan() -> None:
    print("Exp5 run plan. Nothing is executed in this stage.\n")
    print("Deng gate:")
    print(command_text([python_exe(), str(SIM), "--method", "NLR", "--validate-deng", "--validate-only", "--out_dir", str(PREFLIGHT_ROOT / "deng_gate")]))
    print("\nLong-run commands:")
    for spec in RUNS:
        assert_snapshot_cadence(spec)
        print(f"\n[{spec.key}] {spec.label}")
        print(f"  final_time={spec.final_time:g}, snapshots={spec.snapshot_steps}")
        print(f"  save_every={spec.save_every}, viz_every={spec.viz_every}, print_every={spec.print_every}, out_dir={spec.out_dir}")
        print("  " + command_text(base_cmd(spec)))


def stage_smoke(force: bool) -> None:
    remove_dir(PREFLIGHT_ROOT, force)
    PREFLIGHT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Running Deng validation gate...")
    run_cmd([python_exe(), str(SIM), "--method", "NLR", "--validate-deng", "--validate-only", "--out_dir", str(PREFLIGHT_ROOT / "deng_gate")])

    for spec in RUNS:
        assert_snapshot_cadence(spec)
        smoke_dir = PREFLIGHT_ROOT / spec.key
        print(f"\nSmoke test: {spec.label}")
        run_cmd(base_cmd(spec, n_time=1, out_dir=smoke_dir, viz_every=1000, save_every=0, print_every=1))
        row = read_last_log_row(smoke_dir)
        flux_s = float(row["flux_s"])
        solve_s = float(row["solve_s"])
        print(f"  solve_s={solve_s:.4g}, flux_s={flux_s:.4g}, CFL={float(row['CFL']):.4g}")
        if spec.key == "linear" and flux_s > 0.1:
            raise RuntimeError(f"Linear-last smoke flux_s={flux_s:.3f}s > 0.1s; stop and inspect before long run.")
        if spec.key == "pou" and flux_s > 0.1:
            raise RuntimeError(f"PoU smoke flux_s={flux_s:.3f}s > 0.1s; stop and inspect before long run.")
        pngs = list((smoke_dir / "figures").glob("*.png"))
        if pngs:
            raise RuntimeError(f"Smoke test wrote PNG(s) despite viz_every=1000: {pngs[:3]}")
        npys = list(smoke_dir.glob("*.npy"))
        if npys:
            raise RuntimeError(f"Smoke test wrote snapshot(s) despite save_every=0: {npys[:3]}")

    print("\nSmoke tests completed. Long runs were not launched.")


def stage_run(force: bool, only: set[str] | None) -> None:
    selected = [spec for spec in RUNS if only is None or spec.key in only]
    for spec in selected:
        assert_snapshot_cadence(spec)
        remove_dir(spec.out_dir, force)
        spec.out_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nLaunching {spec.label}")
        run_cmd(base_cmd(spec))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["plan", "smoke", "run"], default="plan")
    parser.add_argument("--force", action="store_true", help="Remove existing output directories before running smoke/full stages.")
    parser.add_argument("--only", nargs="*", choices=[spec.key for spec in RUNS], help="Only run selected keys in --stage run.")
    args = parser.parse_args()

    if args.stage == "plan":
        stage_plan()
    elif args.stage == "smoke":
        stage_smoke(args.force)
    elif args.stage == "run":
        stage_run(args.force, set(args.only) if args.only else None)


if __name__ == "__main__":
    main()
