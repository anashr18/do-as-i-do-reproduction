# R1 results — retargeting on the 5080, shipped whisking inputs

Run 2026-09-26 on `local_ubuntu` (RTX 5080 16 GB, sm_120, driver 580.173.02). Every number
here is **[M]**:
- raw rows: `r1_results.jsonl` (14 rows)
- table: `r1_summary.md`
- figure: `r1_frames.png`

All three were copied from `~/rig_work/do_as_i_do/` on the box. The per-run trajectories stay
on the box under `do-as-i-do/retargeting/outputs/sharpa/right/whisking/<data_id>/`.

**What was run:** stage 5 only (physics retargeting) of `malik-group/do-as-i-do` at `824591b`.
The inputs are the authors' shipped `scene.xml`, `trajectory_kinematic.npz` and MANO keypoints
for the whisking clip, so every difference from the shipped result comes from stage 5 alone.
Stages 1–4.5 could **not** be rerun from the shipped demo (§6).

**Environment:**
- `retargeting` conda env, Python 3.12
- torch 2.14.0+cu130
- warp-lang 1.10.1 (CUDA toolkit 12.8)
- mujoco 3.4.0
- mujoco_warp @ `5d5a645`, as pinned

**Harness:**
- `setup.sh`: clone + env
- `r1_run.py`: one arm and seed, recording wall time, peak GPU memory (nvidia-smi) and errors
- `ablation.sh`: arms × seeds
- `analyze.py`: table + figure
- `full_pipeline.sh`: stages 1–5, blocked (§6)

**Metric:** the pipeline's own `compute_object_tracking_error`, which gives the mean object
position error and the mean axis-angle rotation error. It is reported two ways:
- **all** frames, which is what the pipeline logs
- **post**-warmup frames only

Success is the paper's: < 10 cm **and** < 0.5 rad.

## 1. The shipped result, rescored

| | Pos (cm) | Rot (rad) | Frames |
|---|---|---|---|
| All frames: what the pipeline logs | 2.39 | 0.106 | 1600 |
| Post-warmup only | 3.77 | 0.169 | 1000 |
| Warmup only | 0.11 | 0.001 | 600 |

Iterations per replan were `2 13 2 8 2 2 6 2 9 2 6 2 2 2 2 2`, a mean of 4.0: **10 of 16
replans stopped at the 2-iteration minimum** that early stopping allows. The 32-iteration cap
in Table 4 is never reached.

## 2. Reproduction: full method, same inputs

| | Ours (4 runs) | Shipped |
|---|---|---|
| Pos, all (cm) | 2.20 ± 0.12 | 2.39 |
| Pos, post (cm) | 3.46 ± 0.20 | 3.77 |
| Rot, all (rad) | 0.107 ± 0.005 | 0.106 |
| Rot, post (rad) | 0.171 ± 0.008 | 0.169 |
| Iterations/replan | 5.06 ± 0.30 | 4.0 |
| Success | 4 / 4 | yes |

**The reproduction holds.**
- The shipped numbers sit within about 1.6 sd of our run-to-run spread on position, and inside
  it on rotation.
- The harness's all-frames number equals the pipeline's own log line exactly (0.022267 m), so
  the rescoring is the pipeline's metric, not a new one.

## 3. Table 3 ablation on this clip (3 seeds each)

The arms are cumulative, as in the paper.

| Arm | Success | Pos all (cm) | Pos post (cm) | Rot all (rad) | Rot post (rad) | Wall (s) | Peak GPU |
|---|---|---|---|---|---|---|---|
| annealed (SPIDER) | **0 / 3** | 45.31 | 45.31 | 0.408 | 0.408 | 5 | 2.2 GB |
| + warmup | **3 / 3** | 2.29 ± 0.03 | 3.58 ± 0.01 | 0.117 ± 0.009 | 0.187 ± 0.014 | 248 ± 53 | 2.2 GB |
| + perturbation | 3 / 3 | 2.30 ± 0.08 | 3.62 ± 0.15 | 0.104 ± 0.005 | 0.166 ± 0.008 | 484 ± 32 | 6.6 GB |
| + transition (full) | 3 / 3 | 2.20 ± 0.12 | 3.46 ± 0.20 | 0.107 ± 0.005 | 0.171 ± 0.008 | 478 ± 23 | 6.6 GB |

(The full row here pools 4 runs: seeds 0, 1, 2 plus the repeat of seed 0.)

**Reading it:**
- **Warmup is the whole difference on this clip, and the baseline fails immediately.**
  - Without warmup, the simulation starts in the reference's closed-grasp pose, which overlaps
    the whisk. The first reward is −14.87, against −0.003 with warmup.
  - The contact solver ejects the object (peak error 1.29 m), and the run terminates at step
    100 of 945 after 4.9 s.
  - The result is identical to about 1e-6 across seeds: the ejection happens before the
    sampled controls matter.
  - This is failure mode (1) of the paper's §3.2, observed directly.
- **Perturbation:**
  - Position does not change.
  - Rotation after warmup drops from 0.187 to 0.166 rad, about −11%, which is 1–2 sd at n = 3.
    That is suggestive, not established.
  - It costs **about 2× the time and 3× the memory** (4096 worlds instead of 1024).
  - The paper reports the same pattern on reconstructed references: success +1 point, "better
    grasps" qualitatively.
- **The transition penalty has no measurable effect here.** The whisk may never make a
  rest/in-hand transition inside the tracked window [I, not checked against the gate masks].
  The paper's +4 points come from clips that do.
- **One clip is not a success rate.** Table 3's 25% → 71% is over 655 references. This table
  reproduces the *direction* of the first step on one clip, and nothing more.

## 4. Two measurement facts that matter for everything later

**(a) The logged error averages in the warmup, which lowers it by about 36%.**
- Warmup prepends 600 frames that repeat reference frame 0 while the object is welded, so their
  error is about 0.1 cm by construction.
- Over all 1600 frames the full method reads 2.20 cm; over the 1000 tracked frames it reads
  3.46 cm. The shipped run shows the same thing: 2.39 vs 3.77 cm.
- Consequence: an "annealed vs +warmup" comparison on the logged metric compares different
  frame sets. On a clip near 10 cm or 0.5 rad, it can flip a failure into a success.
- Whether the paper's Table 3 used this all-frames mean is not stated [I]. **Future R-stages
  report both.**

**(b) A fixed seed is not reproducible on this GPU.**
- Seed 0 of the full method run twice gave 2.23 / 3.50 cm and 2.16 / 3.37 cm.
- Iterations per replan were 5.25 vs 4.75, and wall time 507 vs 452 s.
- MuJoCo Warp's GPU contact solve is not bit-deterministic, and small differences grow through
  the MPPI selection [I on the mechanism; M on the outcome].
- Consequence: the spread of repeated runs has to be measured like seed spread. One run per
  configuration is not evidence of a difference smaller than about 0.2 cm or 0.01 rad here.

## 5. Cost on the 5080

| | Value |
|---|---|
| One-off Warp kernel compile (first run on the box) | ~2.5 min, cached in `~/.cache/warp` |
| Full method, one clip (1600 frames = 3 s warmup + 4.7 s video) | **478 ± 23 s** |
| Warmup only (1024 worlds) | 248 ± 53 s |
| Seconds per optimizer iteration, 4096 worlds × 600 steps | ~4 s, at 96% GPU utilisation |
| Peak GPU memory, full method | **6.6 GB of 16**, so the 5080 is enough for retargeting |
| Whole R1 session, 14 runs | about 75 GPU-minutes, $0 |

Early stopping sets the cost, and the iteration cap does not: about 5 iterations per replan
against a cap of 32. The worst-case estimate in READING §5.4 overstated the cost by about 8× and
has been corrected.

**A data-generation cost number:** about 8 GPU-minutes on a 5080 to retarget
**4.7 s** of human video once the reconstruction exists. That works out to about 100 GPU-seconds
per second of video, with no reconstruction cost included. R2 adds the reconstruction side.

## 6. What did not run: stages 1–4.5 from the shipped demo

`launch.py --raw-dir ../reconstruction/whisking` fails at stage 1. It needs two files that the
reconstruction pipeline writes and the repo does not ship:
- `config.json`, which can be rebuilt from the README's example
  (`frame_number 125, object whisk, anchor_hand right`)
- `gravity.json`, which comes from GeoCalib in reconstruction stage 2

Running GeoCalib alone here would mix the authors' reconstruction with our gravity estimate. So
the end-to-end stages 1–5 check moves to **R2**, which produces both files. The run failed
before stage 2, so the shared object assets were never touched. `full_pipeline.sh` is ready,
and it restores the authors' assets after itself.

## 7. Open question from the figure

`r1_frames.png`: every run, the authors' included, spikes to about 9–10 cm position error at
**t ≈ 7.72 s**. That is the last frame of the real reference; after it the reference is the last
frame repeated.
- The reference is **not** moving fastest there. Its peak speed is 0.97 m/s at t = 5.8 s, and it
  moves about 11 cm in z within ±0.3 s of the spike.
- Because the spike appears in every independent run, it is a property of the reference or the
  end-of-horizon handling, not optimizer noise [I].
- Two things to test when it matters: the terminal reward (× 10) meeting the padded reference,
  and HaWoR/SAM 3D end-of-clip error.

This is also the frame that decides success on this clip. It passes by 0.5 cm (9.5 cm against
the 10 cm threshold).

## 8. Against PLAN.md R1

| R1 asked for | Result |
|---|---|
| Wall time per clip | 478 ± 23 s (full), after a 2.5 min one-off compile |
| Peak VRAM; do 4096 worlds fit in 16 GB? | 6.6 GB. Yes, with room to spare |
| `E_pos` / `E_rot` against the shipped `.npz` | reproduced within run-to-run spread |
| The ablation on one clip | warmup decisive; perturbation ~−11% rot at 2×/3× cost; transition null here |
| Rerun of `launch.py` (stages 1–5) | blocked by unshipped `config.json` + `gravity.json`, moved to R2 |
