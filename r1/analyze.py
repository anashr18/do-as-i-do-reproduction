"""R1 analysis: summary table per arm + per-frame error figure. Runs on the box in the
`retargeting` env, cwd = do-as-i-do/retargeting (it reuses r1_run's scoring).

    python analyze.py  ->  ~/rig_work/do_as_i_do/r1_summary.md, r1_frames.png
"""

import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from r1_run import ARMS, SHIPPED, WORK, make_config, object_errors, reference_qpos, run_dir, traj_qpos

rows = [json.loads(l) for l in open(WORK / "r1_results.jsonl") if l.strip()]
runs = [r for r in rows if r.get("arm") and r.get("inputs") == "shipped"]

by_arm = defaultdict(list)
for r in runs:
    by_arm[r["arm"]].append(r)


def ms(v, scale=1.0, fmt="{:.2f}"):
    v = np.array(v, dtype=float) * scale
    return fmt.format(v.mean()) + (" ± " + fmt.format(v.std(ddof=1)) if len(v) > 1 else "")


lines = ["| Arm | n | Success (all / post) | Pos all (cm) | Pos post (cm) | Rot all (rad) | Rot post (rad) | Iters/replan | Wall (s) | Peak GPU (MiB) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for arm in ARMS:
    rs = by_arm.get(arm, [])
    if not rs:
        continue
    lines.append(
        f"| {arm} | {len(rs)} | {sum(r['success_all'] for r in rs)} / {sum(r['success_post'] for r in rs)} "
        f"| {ms([r['pos_all'] for r in rs], 100)} | {ms([r['pos_post'] for r in rs], 100)} "
        f"| {ms([r['rot_all'] for r in rs], 1, '{:.3f}')} | {ms([r['rot_post'] for r in rs], 1, '{:.3f}')} "
        f"| {ms([r['mean_opt_iters'] for r in rs])} | {ms([r['wall_s'] for r in rs], 1, '{:.0f}')} "
        f"| {max(r['gpu_peak_mib'] for r in rs)} |"
    )
lines.append("")
lines.append("Per run: " + "; ".join(
    f"{r['arm']} s{r['seed']} (id {r['data_id']}): {r['pos_all']*100:.2f}/{r['pos_post']*100:.2f} cm, "
    f"{r['rot_all']:.3f}/{r['rot_post']:.3f} rad, {r['mean_opt_iters']} it, {r['wall_s']} s"
    for r in runs))
(WORK / "r1_summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# Per-frame object position error: shipped vs our full-method runs vs one warmup-only run.
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ref, cfg = reference_qpos(make_config(99, 0, {}))
q, _ = traj_qpos(SHIPPED / "trajectory_mjwp.npz")
_, pos_s, rot_s = object_errors(q, ref, cfg.warmup_steps)
t = np.arange(len(pos_s)) * 0.005
ax[0].plot(t, pos_s * 100, color="k", lw=1.6, label="shipped (authors)")
ax[1].plot(t, rot_s, color="k", lw=1.6)
for r in runs:
    if r["arm"] not in ("full", "warmup") or r["n_frames"] < 1000:
        continue
    f = np.load(run_dir(r["data_id"]) / "r1_err_frames.npz")
    style = dict(lw=0.9, alpha=0.8, color="tab:blue" if r["arm"] == "full" else "tab:orange")
    ax[0].plot(t[: len(f["pos"])], f["pos"] * 100, label=f"{r['arm']} s{r['seed']} (id {r['data_id']})", **style)
    ax[1].plot(t[: len(f["rot"])], f["rot"], **style)
for a in ax:
    a.axvspan(0, cfg.warmup_steps * 0.005, color="0.9", zorder=0)
ax[0].axhline(10, color="tab:red", ls="--", lw=0.8)
ax[1].axhline(0.5, color="tab:red", ls="--", lw=0.8)
ax[0].set_ylabel("object pos error (cm)")
ax[1].set_ylabel("object rot error (rad)")
ax[1].set_xlabel("sim time (s); grey = warmup (object welded, reference = frame 0)")
ax[0].legend(fontsize=7, ncol=2)
ax[0].set_title("Do as I Do retargeting, whisking clip, RTX 5080 — per-frame object error")
fig.tight_layout()
fig.savefig(WORK / "r1_frames.png", dpi=130)
print("wrote", WORK / "r1_frames.png")
