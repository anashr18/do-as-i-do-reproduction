# R2 results — the reconstruction pipeline on the shipped whisking clip, end to end

Run 2026-09-26 on a RunPod **RTX PRO 6000 Blackwell Server Edition** (96 GB, sm_120, driver
580.159.04), secure cloud, $2.09/hr. No RTX 5090 was available on either cloud. **Total:
120.2 min, $4.19**. Numbers are **[M]** unless tagged.

**What was run:**
- `reconstruction/run_pipeline.sh` at `824591b`, stage for stage and flag for flag
  (`run_recon.sh`), on the authors' whisking clip.
- A comparison against the authors' shipped reconstruction of that clip (`compare.py`).
- All five retargeting stages on **our** reconstruction (`retarget_ours.sh`). That closes the
  loop from raw video to physics-retargeted hand trajectory, and it is the stages 1–5 check
  R1 could not run.

**Artifacts** (`artifacts/`, 12 MB; `.npz` files are gitignored):
- `stage_metrics.tsv`, the per-stage GPU memory traces, `r2_recon.log`
- our layouts, mesh, masks, gravity and TAPIR motion stats
- `mask125.png`
- `r2_compare.json`
- `retarget_300/` (scene, trajectories, config, log)

**Deviations, recorded in `run_recon.sh`:**

| | Deviation | Consequence |
|---|---|---|
| D1 | `ffmpeg -vsync 0` → `-fps_mode passthrough` | none; the option was removed in ffmpeg 9 (§1) |
| D2 | SAM 3 object prompt as two points (566,486; 582,525) instead of a GUI click | our object masks differ from the authors' in an unknown way; theirs were not shipped |
| D3 | **HaWoR not run** (no MANO licence). The authors' `all_hand_meshes.npz` is used | **the hand, and therefore the metric scale, is theirs, not ours.** The k agreement in §4 is a consistency check, not an independent one |
| D4 | fresh run directory | none |

---

## 1. Four things that break, or would silently change results, on a fresh build today

1. **ffmpeg 9 breaks two stages.**
   - `01_create_envs.sh` installs `ffmpeg` unpinned, and on 2026-09-26 that resolved to 9.0.2.
   - ffmpeg 9 removed `-vsync`, which `run_pipeline.sh` passes in stages 0 and 1. mediapy's
     video reader then fails inside TAPIR ("Unable to parse video header").
   - The authors' exported envs have ffmpeg 8.0.0/8.0.1. The fix was D1, plus pinning
     `ffmpeg=8.0.0` and `mediapy==1.2.6` in the tapnet env.
2. **`02_fetch_weights.sh --download` would overwrite Fast-SAM3D's tracked configs.**
   - It rsyncs Hugging Face's checkpoint configs into both submodules' `checkpoints/hf/`.
   - Fast-SAM3D tracks an edited `pipeline.yaml` and a `slat_generator_faster.yaml` there in git.
   - The rsync would replace the fork's pipeline config with upstream's and say nothing.
   - `fetch_weights.sh` uses `cp -n`, and verified that the tracked configs were unchanged.
3. **SAM 3D's texture baking needs mip-splatting's `diff-gaussian-rasterization`.** Stage 2
   died with `NameError: GaussianRasterizationSettings` after 146 s, as `env/README.md` warns.
   It was built with `TORCH_CUDA_ARCH_LIST=12.0`.
4. **`LIDRA_SKIP_INIT=true`** must be set before importing `sam3d_objects`. Upstream sets it
   inside its entry points, so a bare import fails. It was set as a conda env var.

**The SAM 3D env on Blackwell** (`pod_setup.sh`), built to match `env/sam3d.yml`:
- Python 3.11, torch 2.8.0+cu128, and the export's pinned pip set.
- The fork's pinned pytorch3d / gsplat / MoGe commits, built for sm_120. The pytorch3d commit
  reports 0.7.8, where the export says 0.7.9.
- kaolin 0.18.0 and nvdiffrast.
- Build times: base 480 s, pytorch3d 223 s, everything else under 75 s each.
- Weights (SAM 3D 13 GB + SAM 3 + MoGe-vitl + BootsTAPIR): 229 s.

## 2. Stage timings and peak GPU memory (one 138-frame clip, 1280×720)

| Stage | Wall | Peak GPU |
|---|---|---|
| s0 frames | 2.7 s | — |
| s1 SAM 3 object (points) | 178 s | 8.9 GB |
| s1 SAM 3 hand (text) | 163 s | 8.9 GB |
| s2 SAM 3D mesh, frame 125 | 188 s | **23.4 GB** |
| s2 MoGe, ref frame / all frames | 32 s / 76 s | 3.6 GB |
| s2 GeoCalib gravity | 254 s | 1.4 GB |
| s2.5 BootsTAPIR | 82 s | 9.7 GB |
| **s3 guided-diffusion tracking** | **3695 s (61.6 min)**, 25.7 s/frame after a 2 min first frame | 22.5 GB |
| s3 projection / camera frame / s4 scale | 49 s / 0.4 s / 9 s | 3.1 GB |
| **Reconstruction total** | **≈ 79 min** | **23.4 GB** |
| Retargeting, stages 1–5 (fresh pod, incl. ~3 min Warp compile) | 798 s | 6.9 GB |

What this means for hardware:
- **The peak is 23.4 GB** (SAM 3D mesh generation), under the README's stated 32 GB minimum.
  On this clip the reconstruction would fit a 24 GB card, but not the 16 GB 5080 [M on the
  peak, I on other clips].
- **Tracking is 78% of reconstruction time.** It is the stage R6 would attack first:
  25 samples × 25 Euler steps per frame, plus per-frame render-and-compare.
- **Data-generation cost:** about 80 GPU-minutes on a $2.09/hr card to reconstruct 4.6 s of video, then about
  8–13 min to retarget it. That is **about $3 of rented GPU per 4.6 s clip**, or roughly **$0.65
  per second of video**, with HaWoR excluded.

## 3. Stage checks

- **SAM 3 masks:**
  - Object point prompt: score 1.00. Hand text prompt: 0.89. Both track 138 frames, the same
    count as the shipped layout.
  - The whisk mask includes a small blob of milk and pan rim at the wire tip (`mask125.png`).
- **GeoCalib:** roll 0.93°, pitch −5.86°, 128/138 inlier frames.
- **TAPIR rotation speed:** median 2.0°/frame, max 22.6°/frame. 12 of 137 frames are faster
  than 6.7°/frame, which puts α_p at its 0.1 floor.
- **Render-IoU against the SAM 3 mask:** mean 0.31, and under 0.1 on 10 frames. That is low,
  but the whisk is thin, and (§4) **IoU does not predict where the pose is wrong** (rank
  correlation −0.03 with orientation disagreement). As a per-frame quality signal it is weak
  for thin objects.

## 4. Our reconstruction vs the authors' (`r2_compare.json`)

| | Ours | Shipped |
|---|---|---|
| Frames tracked | 138 | 138 |
| Mesh extent (cm), after metric scale | 6.22 × 17.88 × 5.69 | 6.09 × 18.05 × 5.94 |
| Mesh vertices | 5502 | 5068 |
| SAM 3D raw scale → metric scale | 0.1617 → 0.1787 | 0.1637 → 0.1809 |
| k at anchor frame · clip mean · CV | 1.1050 · 1.0832 · 2.58% | 1.1050 · 1.0832 · 2.58% |
| Chamfer at anchor frame, camera space | **0.41 cm** | |
| Per-frame Chamfer, camera space (median / p90 / max) | **0.57 / 1.21 / 3.05 cm** | |
| Translation difference (median / p90 / max) | **0.95 / 2.24 / 4.51 cm** | |
| of which depth (z) / lateral (xy), median | 0.45 / 0.71 cm | |
| Rotation difference (median / p90 / max) | **19.4° / 56.1° / 93.6°** | |
| Handle-axis direction difference (median / p90 / max) | 9.2° / 23.0° / 81.9° | |
| Path jitter, 2nd difference (median) · depth share | 3.0 cm · 88% | 3.3 cm · 89% |

**Reading it:**
- **Shape and position reproduce.** The mesh is within about 1–4% per axis, the Chamfer
  distance is 0.4–0.6 cm, and the object path is within about 1 cm (median).
- **k is identical to four decimals.** Under D3 that is expected: k depends only on the hand
  (the same shipped HaWoR) and MoGe (deterministic). It does show that our MoGe run and hand
  masks reproduce theirs. It is **not** independent evidence about scale.
- **Orientation does not reproduce, and the disagreement grows with distance from the
  anchor frame.**

  | Frames | Rotation diff (median) | Translation diff (median) |
  |---|---|---|
  | anchor 125 | 7.6° | 0.34 cm |
  | 1–40 before the anchor | 13.8° | 0.71 cm |
  | 40–85 before | 17.0° | 0.93 cm |
  | **85–125 before** | **44.0°** | 1.08 cm |
  | after the anchor (126–137) | 20.1° | 1.14 cm |

  - Rank correlation of |frame − 125| with rotation difference: 0.35.
  - About half the total rotation difference is spin about the handle, which a whisk barely
    shows. The rest is a real tilt of the handle: 9° median, 23° p90.
  - The tracker **chains** each frame's pose from the previous one, and the release's anchor
    (frame 125 of 138) forces 125 frames of backward chaining. Orientation error accumulates
    along the chain. Translation hardly does, because the scale step re-anchors it to the hand
    on every frame (§3 of READING).
  - Candidate remedy [I]: put the anchor in the middle of the clip, which halves the longest
    chain.
  - What we cannot separate here: seed/GPU nondeterminism, our point prompt vs their click
    (D2), and pytorch3d 0.7.8 vs 0.7.9. R1 already showed a fixed seed is not reproducible on a
    GPU.
- **Depth dominates the jitter in both runs** (88–89%): READING §3's prediction, now shown
  twice.

## 5. End to end: retargeting our own reconstruction (stages 1–5)

| | Ours, end to end | Authors' shipped (R1 §1) |
|---|---|---|
| Object error, all frames | 2.69 cm / 0.173 rad | 2.39 cm / 0.106 rad |
| Object error, post-warmup | 4.26 cm / 0.276 rad | 3.77 cm / 0.169 rad |
| Worst post-warmup frame | 12.4 cm / 0.71 rad | 10.2 cm / 0.50 rad |
| Success (< 10 cm and < 0.5 rad, mean) | yes / yes | yes / yes |
| Iterations per replan | 5.56 | 4.0 |

- **The pipeline runs from raw video to a successful physics retarget on our build.** Stages 1–4.5
  took about 30 s; stage 5 took about 7 min of optimisation after the Warp compile.
- **Rotation error is 63% higher** than on the authors' reconstruction (post-warmup 0.276 vs
  0.169 rad). That fits §4: the retargeter tracks whatever orientation the tracker produced,
  and ours drifted more far from the anchor.
- Stage 1's spike cleaner flagged **22 rotation frames and 7 position frames** in our
  reference.
- **This settles an open question from R1.** The in-hand gate reads the whisk as held on
  **100% of frames** (2245/2245), so the transition penalty's rest branch can never fire on
  this clip. That is why R1 measured no effect from it.
- The perturbation gate is active on 73% of frames (every post-warmup frame).

## 6. Cost

| | |
|---|---|
| Pod | RTX PRO 6000 Blackwell, secure, $2.09/hr |
| Up | 120.2 min |
| **Cost** | **$4.19** |
| Of which | setup + weights ≈ 25 min · debugging the four breakages ≈ 10 min · reconstruction ≈ 79 min · retarget ≈ 13 min |
| Next time | a warm rerun would skip about 25 min of setup; the four fixes are now in `pod_setup.sh` |

The HF token was copied to the pod at the operator's choice, and deleted from the pod before
termination.
