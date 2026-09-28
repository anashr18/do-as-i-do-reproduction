"""R1 harness: run Do-as-I-Do's physics retargeting (stage 5) on the shipped whisking inputs.

Runs on the 5080 inside the `retargeting` env, with cwd = do-as-i-do/retargeting.

    python r1_run.py run   --arm full --seed 0 --data-id 10 --out r1_results.jsonl
    python r1_run.py score --npz PATH/trajectory_mjwp.npz --data-id 10 --warmup 1

`run` copies the authors' scene.xml + trajectory_kinematic.npz (from the untouched
shipped_outputs snapshot) into outputs/sharpa/right/whisking/<data-id>/, so stage 5 sees
exactly the inputs the shipped result was made from. Stages 1-4.5 are not rerun here.

Arms are Table 3's cumulative rows. "annealed" = SPIDER's annealed sampling with the three
additions switched off; everything else is the release's do_as_i_do.yaml.

Error is the pipeline's own compute_object_tracking_error: mean object position error (m)
and mean axis-angle rotation error (rad) over every simulated frame. It is reported twice:
over all frames (what the pipeline logs), and over post-warmup frames only, because warmup
prepends `warmup_steps` frames whose object reference is frame 0 while the object is welded.
Success follows the paper: E_pos < 0.1 m and E_rot < 0.5 rad.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

WORK = Path.home() / "rig_work" / "do_as_i_do"
SHIPPED = WORK / "shipped_outputs" / "sharpa" / "right" / "whisking" / "0"
# The in-hand gate (transition penalty, perturbation gate) reads the MANO keypoints per data_id.
SHIPPED_MANO = WORK / "shipped_outputs" / "mano" / "right" / "whisking" / "0"
TASK, ROBOT, SIDE = "whisking", "sharpa", "right"
POS_OK, ROT_OK = 0.1, 0.5

OFF_PERTURB = dict(num_perturb_samples=1, perturb_force_scale=0.0, perturb_torque_scale=0.0)
ARMS = {
    "annealed": dict(warmup_duration=0.0, pedestal_penalty_scale=0.0, **OFF_PERTURB),
    "warmup": dict(pedestal_penalty_scale=0.0, **OFF_PERTURB),
    "perturb": dict(pedestal_penalty_scale=0.0),
    "full": dict(),
}


class GpuPeak(threading.Thread):
    """Poll nvidia-smi for device memory used; keeps the max. The box is otherwise idle."""

    def __init__(self, period_s: float = 0.5):
        super().__init__(daemon=True)
        self.period_s, self.peak, self.stop_flag = period_s, 0, False

    def sample(self) -> int:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        ).stdout.strip().splitlines()
        return int(out[0]) if out else 0

    def run(self):
        while not self.stop_flag:
            self.peak = max(self.peak, self.sample())
            time.sleep(self.period_s)


def run_dir(data_id: int) -> Path:
    return Path("outputs") / ROBOT / SIDE / TASK / str(data_id)


def stage_inputs(data_id: int) -> None:
    dst = run_dir(data_id)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("scene.xml", "trajectory_kinematic.npz"):
        shutil.copy2(SHIPPED / name, dst / name)
    mano_dst = Path("outputs") / "mano" / SIDE / TASK / str(data_id)
    mano_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SHIPPED_MANO / "trajectory_keypoints.npz", mano_dst / "trajectory_keypoints.npz")


def make_config(data_id: int, seed: int, overrides: dict):
    from launch import load_mjwp_config

    return load_mjwp_config(
        dataset_name="do_as_i_do", task=TASK, data_id=data_id, robot_type=ROBOT,
        embodiment_type=SIDE, seed=seed, wait_on_finish=False, max_sim_steps=0,
        force=True, show_viewer=False, **overrides,
    )


def object_errors(qpos_traj: np.ndarray, qpos_ref: np.ndarray, warmup_steps: int) -> dict:
    from retargeting.utils.tracking_error import compute_object_tracking_error

    e = compute_object_tracking_error(qpos_traj, qpos_ref[: len(qpos_traj)], SIDE, "mjwp")
    pos_f, rot_f = e["pos_err_frames"], e["quat_err_frames"]
    post = slice(warmup_steps, None)
    r = dict(
        n_frames=int(len(pos_f)), warmup_steps=int(warmup_steps),
        pos_all=float(pos_f.mean()), rot_all=float(rot_f.mean()),
        pos_post=float(pos_f[post].mean()), rot_post=float(rot_f[post].mean()),
        pos_warm=float(pos_f[:warmup_steps].mean()) if warmup_steps else None,
        rot_warm=float(rot_f[:warmup_steps].mean()) if warmup_steps else None,
        pos_max_post=float(pos_f[post].max()), rot_max_post=float(rot_f[post].max()),
    )
    r["success_all"] = bool(r["pos_all"] < POS_OK and r["rot_all"] < ROT_OK)
    r["success_post"] = bool(r["pos_post"] < POS_OK and r["rot_post"] < ROT_OK)
    return r, pos_f, rot_f


def reference_qpos(config) -> np.ndarray:
    """The object reference exactly as stage 5 builds it (interpolated, warmup-prepended)."""
    from retargeting.config import process_config
    from retargeting.utils.io import load_data

    config = process_config(config)
    return load_data(config, config.data_path)[0].detach().cpu().numpy(), config


def traj_qpos(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    q = d["qpos"]
    opt_steps = d["opt_steps"] if "opt_steps" in d.files else np.array([])
    return q.reshape(-1, q.shape[-1]), opt_steps


def cmd_run(a) -> None:
    from retargeting.pipeline.optimize_physics import main as optimize_physics

    stage_inputs(a.data_id)
    config = make_config(a.data_id, a.seed, ARMS[a.arm])
    base = GpuPeak().sample()
    mon = GpuPeak()
    mon.start()
    t0 = time.perf_counter()
    logged = optimize_physics(config)
    wall = time.perf_counter() - t0
    mon.stop_flag = True
    mon.join()

    ref, cfg = reference_qpos(make_config(a.data_id, a.seed, ARMS[a.arm]))
    q, opt_steps = traj_qpos(run_dir(a.data_id) / "trajectory_mjwp.npz")
    err, pos_f, rot_f = object_errors(q, ref, cfg.warmup_steps)
    np.savez(run_dir(a.data_id) / "r1_err_frames.npz", pos=pos_f, rot=rot_f)
    row = dict(
        arm=a.arm, seed=a.seed, data_id=a.data_id, inputs="shipped",
        wall_s=round(wall, 1), gpu_peak_mib=mon.peak, gpu_base_mib=base,
        num_samples=cfg.num_samples, num_perturb_samples=cfg.num_perturb_samples,
        worlds=cfg.num_samples * cfg.num_perturb_samples,
        replans=int(len(opt_steps)), mean_opt_iters=float(np.mean(opt_steps)) if len(opt_steps) else None,
        logged_pos=float(logged["obj_pos_err"]) if logged else None,
        logged_rot=float(logged["obj_quat_err"]) if logged else None,
        **err,
    )
    print(json.dumps(row))
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")


def cmd_score(a) -> None:
    if not a.keep_inputs:
        stage_inputs(a.data_id)
    overrides = {} if a.warmup else ARMS["annealed"]
    ref, cfg = reference_qpos(make_config(a.data_id, 0, overrides))
    q, opt_steps = traj_qpos(Path(a.npz))
    err, _, _ = object_errors(q, ref, cfg.warmup_steps)
    row = dict(scored=str(a.npz), replans=int(len(opt_steps)),
               mean_opt_iters=float(np.mean(opt_steps)) if len(opt_steps) else None, **err)
    print(json.dumps(row))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=list(ARMS), required=True)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--data-id", type=int, required=True)
    r.add_argument("--out", default=str(WORK / "r1_results.jsonl"))
    s = sub.add_parser("score")
    s.add_argument("--npz", required=True)
    s.add_argument("--data-id", type=int, default=99)
    s.add_argument("--warmup", type=int, default=1)
    s.add_argument("--keep-inputs", action="store_true",
                   help="score against the data-id dir's own scene/reference (a full-pipeline run)")
    s.add_argument("--out", default="")
    a = p.parse_args()
    {"run": cmd_run, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
