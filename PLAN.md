# Do as I Do — replication plan

Written 2026-09-25, before any stage ran; each stage is marked where it closed. Read
`READING.md` first. Every stage ends in a number, a diagnosed failure or a merged PR, and a
result is quoted only once a run has produced it.

Stages are ordered by **what can start with no permissions and no purchase**. Each one closes
on a number. Each stage closes before the next one starts.

## R0 — Access (accounts and licences; approvals take days)

Each one needs an account and acceptance of its licence.

| What | Where | Needed by |
|---|---|---|
| Request access to `facebook/sam-3d-objects` | huggingface.co/facebook/sam-3d-objects | R2 |
| Request access to `facebook/sam3` | huggingface.co/facebook/sam3 | R2 |
| Register and download `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` | mano.is.tue.mpg.de | R2 (HaWoR) |
| DexYCB (CC BY-NC 4.0) and HOI4D (request form) | their project pages | R3 |
| OakInk2 | its project page | R3b |

**R1 needs none of these.**

## R1 — Retargeting on the shipped demo (5080, no gated weights)

> **Done 2026-09-26. See `r1/RESULTS.md`.**
> - The reproduction matches the authors' shipped result within run-to-run spread.
> - One clip costs 478 s and 6.6 GB on the 5080.
> - Warmup decides success; the annealed baseline fails 0/3.
> - The logged error includes the warmup frames, which lowers it by about 36%.
> - A fixed seed is not reproducible on this GPU.
> - Stages 1–4.5 from the shipped demo are blocked by the unshipped `config.json` and
>   `gravity.json`, so that check moves to R2.

The repo ships the whisking clip's reconstruction output (hand meshes, optimised layout, masks)
**and** the authors' retargeted result.
1. Build the `retargeting` env on the 5080: Python 3.12, `mujoco==3.4.0`, `warp-lang==1.10.1`,
   and mujoco_warp pinned to `5d5a645`.
2. Replay the shipped `trajectory_mjwp.npz` in viser, as the known-good reference.
3. Re-run `launch.py --task whisking --raw-dir ../reconstruction/whisking --no-show-viewer`.
4. Run the four Table 3 configurations on this clip, with 3 seeds each: annealed only
   (`warmup_duration 0`, `num_perturb_samples 1`, `pedestal_penalty_scale 0`), then add each
   addition back.

**Numbers:**
- **[M]** wall-clock time per clip
- **[M]** peak VRAM: does 1024 × 4 = 4096 worlds fit in 16 GB?
- `E_pos` / `E_rot` compared with the shipped `.npz`
- how the ablation moves on one clip

If it runs out of memory, halve `num_samples` and record the quality cost, or use a RunPod 5090.

## R2 — Reconstruction on the shipped clip (RunPod, a card with ≥ 32 GB; start at 48 GB)

> **Done 2026-09-26. See `r2/RESULTS.md`. $4.19, 120 min, RTX PRO 6000 Blackwell; HaWoR was
> replaced by the shipped hand output, pending MANO.**
> - The pipeline ran from raw clip to a successful physics retarget.
> - Peak GPU memory was 23.4 GB. Tracking took 61.6 min, 78% of the reconstruction time.
> - Shape and position reproduce the authors' run: 0.4–0.6 cm Chamfer, 0.95 cm median path
>   difference.
> - **Orientation does not reproduce:** the difference is 7.6° at the anchor frame and 44° at
>   85–125 frames from it, because the tracker chains poses.
> - End-to-end rotation error is 63% higher than on the authors' reconstruction.
> - Four release breakages were fixed on the way: ffmpeg 9, the config overwrite in the fetch
>   script, the rasterizer, and `LIDRA_SKIP_INIT`.
> - R1's null transition effect is explained: the whisk is in hand on 100% of frames.

1. Build the four envs from the `env/*.yml` cu128 pins, which support Blackwell.
2. SAM 3's object click needs an X display. On the pod, use `--text` prompts or X forwarding.
3. Run `run_pipeline.sh whisking/whisking.mp4 125 whisk right`, diff the result against the
   shipped `layout_camera_frame_optimized.json`, then feed our own output through R1's
   retargeting to close the loop end to end. This run also writes `config.json` and
   `gravity.json`, which `r1/full_pipeline.sh` needs for the stages 1–5 check that R1
   could not run.
4. Report errors both over all frames and after warmup only (R1 §4a). Run repeats as well as
   seeds, because a fixed seed is not reproducible (R1 §4b).

**Numbers:**
- **[M]** time per stage
- **[M]** peak VRAM per stage: which stages would fit the 5080?
- pose difference against the shipped output
- end-to-end success on our own reconstruction

## R3 — The paper's tables

- **R3a, Tables 2 and 5.** Write the evaluation harness, which is not in the release:
  object-level F-5, F-10 and Chamfer distance on DexYCB (160 clips) and HOI4D (12), with ground-truth
  hands, following MCC-HO and G-HOP. Ablations:
  - adaptive vs fixed α_p
  - clustering vs random selection
  - **`--post_optimize` off**. This is the one the paper doesn't run (READING §4.4).
- **R3b, Table 3 right half.** Run OakInk2 through upstream SPIDER's `oakinkv2` config, with the
  three additions ported onto it.

**Numbers:** our table next to theirs, with every gap explained or recorded as a failure.

## R4 — The absolute-depth experiment

The paper treats the hand as a metric ruler but never measures the metric error on real clips
(READING §3). Measure it:
- Record about 20 clips of hand-object manipulation on a desk, with **one RGB camera as the input**
  and **ground truth from an ArUco tag of known size**. The ground truth can come from the same
  camera, calibrated, or from a second camera. No depth sensor is required.
- Run the pipeline on RGB only.

**Numbers:**
- object translation error in cm
- scale error in %
- depth error along the viewing ray vs across it (the paper's contact/occlusion limitation)
- how much per-frame `k` jitter the disabled `smooth_trajectory.py` removes

## R5 — The playbook funnel on 100DOH

Sample about 300 clips and reproduce the 9% → 4% funnel. Add the column the paper leaves out:
**GPU-minutes per surviving trajectory**. This is the first end-to-end **cost/yield number for data generation**.

## R6 — Data-generation efficiency

Map the quality–cost Pareto front over:
- the number of pose samples (25 → 4)
- Euler steps (25 → 8; this is `few_step` territory)
- Taylor cache on/off
- MPPI `num_samples` (1024 → 128)
- perturbation replicas K (4 → 1)

**Number:** the same success rate at a stated fraction of the compute, or the knee where it
falls off.

## R7 — A low-cost embodiment: the SO-101

MuJoCo Menagerie ships `robotstudio_so101`. Retarget MANO onto the SO-101: thumb–index
aperture drives the gripper, and the wrist pose is constrained by a 5-DoF arm. This is **not a
config change**, because the release floats a 6-DoF wrist and solves the arm afterwards. For
the SO-101, reachability has to move **inside** the MPC. Then replay on the real arm open-loop,
and train ACT on the generated trajectories against teleop data.

**Numbers:**
- retarget success on the SO-101
- real replay success
- policy success with and without video-derived data

## Workshop-paper candidates (not now)

Each of these produces a result the paper does not report:
1. **R4**: the metric error of hand-anchored monocular reconstruction, measured in the wild.
2. **R6**: "Do as I Do at a fraction of the compute", an efficiency Pareto for video-to-robot
   data generation.
3. **R7**: human video to a $250 arm, with a trained policy. 

## Hardware — prices web-checked 2026-09-25 (≈ ₹96/USD); re-check before buying

**Replicating the paper needs no purchase.** The input is monocular RGB. The Sharpa model is
public (`sharpa-robotics/sharpa-urdf-usd-xml`, and Menagerie's `sharpa_wave`). R1–R6 run in
simulation plus RunPod. Only the paper's Fig. 6 (real Sharpa + UR3e) needs the paper's hardware,
and a Sharpa Wave is about **$50k, sold to companies only**, with India shipping unconfirmed.
That puts it out of range and makes it the wrong target.

**Hands SPIDER already supports**, meaning no porting: ability, allegro, inspire, metahand,
schunk, xhand. LEAP, the Amazing Hand, ORCA and the SO-101 each need a port, following
`docs/development/add-robot.md` in SPIDER. Note that SPIDER is **CC BY-NC 4.0**.

| Tier | What | Landed cost (with ~25–30% duty + GST) | Buy when |
|---|---|---|---|
| **₹0** | nothing; RunPod for R2 and R3 | pod time only | **now** |
| **~₹15k** | Amazing Hand kit, $99 (Seeed): 8 DoF, 4 fingers, 400 g. The community has mounted it on an SO-101 with the gripper servo removed | ~₹13–15k | after R1–R3 pass. It is a cheap real dexterous target for R7, but 400 g is close to the SO-101's unofficial ~500 g payload |
| **~₹5L** | AgileX Piper, ₹2.45L at MGSL (India stock, 6 DoF, 1.5 kg payload), plus a LEAP v1 hand (16 DoF, ROBOTIS XC330 bundle $1,839; servos out of stock in India) | ~₹4.8–5L | only if R7 on the SO-101 shows **the arm, not the method**, is the limit. It follows a full 6-DoF wrist, and LEAP is the cheapest real anthropomorphic stand-in for Sharpa |
| **~₹12–21L** | OpenArm 2.0 two-arm set ($6–6.5k, WowRobo, mainline LeRobot) plus 2× LEAP (~₹12L), or 2× Inspire RH56DFX, already in SPIDER (~₹21L) | ~₹12–21L | defer. This matches the paper's two-arm setup, but only earns its cost after a policy has run on the SO-101 |

Unverified figures stay out of this table: XHand1, Ability, Linker, Dex5, and used UR3e/FR3
prices are reference numbers or search snippets. The Aero Hand store says it is no longer
selling.
