# Do as I Do — a code-verified reading

**Paper:** *Do as I Do: Dexterous Manipulation Data from Everyday Human Videos* —
Paliwal\*, Etukuru\*, Liang\*, Abbeel, Shafiullah, Malik (UC Berkeley).
arXiv [2606.19333](https://arxiv.org/abs/2606.19333) v1, 17 Jun 2026. Project page do-as-i-do.com.
**Code:** [malik-group/do-as-i-do](https://github.com/malik-group/do-as-i-do), MIT, read at
`824591b` (2 Aug 2026). The object tracker is in the submodule fork
[malik-group/Fast-SAM3D](https://github.com/malik-group/Fast-SAM3D) at `823d478`.

Read 2026-09-25. Evidence tags:

- **[P]**: stated in the paper (section, table or equation given).
- **[S]**: read from the released source, with `file:line` at the commits above.
- **[I]**: inferred, and not verified. Confirm it before repeating it as fact.
- **[M]**: measured by this replication. The runs are in `r1/RESULTS.md` (retargeting on the 5080,
  2026-09-26) and `r2/RESULTS.md` (reconstruction plus end to end on RunPod, 2026-09-26).

---

## 0. The paper in one paragraph

The input is a monocular RGB clip of one human hand manipulating one rigid object: an internet
video, an egocentric recording, or the output of a video generator. The output is a 22-DoF
Sharpa Wave hand, its 6-DoF wrist and the object moving together in MuJoCo, physically
consistent, and replayable open-loop on a real UR3e + Sharpa hand at 50 Hz. There are two
halves. **Reconstruction** (§3.1) uses foundation models: SAM 3 for masks, SAM 3D for the object
mesh, MoGe-2 for depth, HaWoR for the hand, BootsTAPIR for point tracks, GeoCalib for gravity.
Its one new algorithm is a way to make SAM 3D track the object through a video. **Retargeting**
(§3.2) is SPIDER's sampling-based MPC in MuJoCo Warp, plus three additions that make it
tolerate *noisy* references: warmup, random force perturbation and a contact-transition
penalty. The headline numbers are 71% retargeting success on in-the-wild references (25% for
plain SPIDER), and a 67% / 18% / 15% human preference over FoundationPose for object tracking.
The paper also finds that only 4% of pre-filtered internet clips survive the pipeline. [P]

## 1. SAM 3 vs SAM 3D

These are two different models, and the paper uses both. [P §3.1] [S `run_pipeline.sh`]

| Model | What it does here | Stage |
|---|---|---|
| **SAM 3** ("Segment Anything with Concepts") | Video masks. The object comes from a **click** on a reference frame, the hand from a **text** prompt (`"right hand"`). | 1 |
| **SAM 3D Objects** ("3Dfy anything") | From one image plus a mask, it produces the object's **mesh and its pose** (the "layout") in the camera frame. | 2, 3 |

The paper's one new vision idea is to turn SAM 3D, a single-image generator, into a
**video tracker** without retraining it (§3 below). SAM 3 only supplies masks.

## 2. Pipeline, stage by stage

The pipeline uses four conda envs because the models' dependency pins conflict. [S `reconstruction/README.md`]

| # | Stage | Code | Env |
|---|---|---|---|
| 0 | ffmpeg frame extraction | `run_pipeline.sh:179` | sam3 |
| 1 | SAM 3 masks: object by click, hand by text | `scripts/run_sam3_video.py` | sam3 |
| 2 | SAM 3D mesh from the reference frame | `modules/sam-3d-objects/generate_mesh_sam3d.py` | sam3d |
| 2 | MoGe pointmaps + intrinsics (reference frame, then every frame) | `scripts/get_pointmap_dir.py` | sam3d |
| 2 | HaWoR hands (run with `--static_camera`, focal length from MoGe) | `modules/HaWoR/demo.py` | hawor |
| 2 | Gravity direction (GeoCalib) | `scripts/predict_video_gravity.py` | sam3d |
| 2.5 | BootsTAPIR point tracks → per-frame object rotation speed | `scripts/tapir_velocity_tracking.py` | tapnet |
| 3 | **Guided-diffusion object tracking** | `modules/Fast-SAM3D/track_object.py` | sam3d |
| 4 | Hand–object scale alignment | `scripts/optimize_translation_scale.py` | sam3d |
| R1–R5 | clean + gravity-align → CoACD → MJCF → mink IK → **MPC in MuJoCo Warp** | `retargeting/launch.py` | retargeting |
| D | IK onto dual UR3e, then stream at 50 Hz | `deployment/` | deployment |

The requirements come from the README. Reconstruction needs an **NVIDIA GPU with ≥ 32 GB VRAM**.
Measured on the whisking clip, the peak is **23.4 GB** (SAM 3D mesh generation) [M], so a 24 GB
card would do for that clip. The 5080's 16 GB still rules it out for that half. Retargeting needs any CUDA GPU. The weights for
SAM 3D and SAM 3 are **gated** on Hugging Face. MANO needs a licence registration. The Sharpa
hand SDK is proprietary and is not shipped. [S]

---

## 3. Absolute depth: how the paper handles metric scale

The paper never measures its metric error on real clips, so this section matters most. The
pipeline produces four
kinds of geometry, and **each has its own scale**:

| Source | What it gives | Scale |
|---|---|---|
| MoGe pointmap | per-pixel 3D, per frame | the paper cites MoGe-2 ("metric"), but **the release loads MoGe v1**, which is not metric (next paragraph) [S]. Either way, the global scale error is large and changes from frame to frame [I] |
| SAM 3D layout | object pose + mesh | expressed in the MoGe pointmap's frame, because SAM 3D was **trained on MoGe pointmaps** [P App. A] |
| HaWoR hand | MANO mesh per frame | "near metric": the MANO shape prior fixes the hand's size [P §3.1] |
| GeoCalib | gravity direction | direction only, no scale |

**The release uses MoGe v1, not MoGe-2** (found 2026-09-26). `reconstruction/scripts/get_pointmap_dir.py:22-35`
loads the depth model from Fast-SAM3D's `checkpoints/hf/pipeline.yaml:61-65`, which is
`moge.model.v1.MoGeModel` / `Ruicheng/moge-vitl`. MoGe's README lists metric scale for MoGe-2
only. So in the released pipeline the pointmap has no trustworthy absolute scale, and the hand is
the *only* metric source. SAM 3D was trained on MoGe pointmaps, which is why the pipeline uses
MoGe at all [P App. A]. Whether the paper's own runs used v1 or v2 is not stated [I].

**The decision the paper makes:** use the **hand** for absolute scale, and use the pointmap
only for the **relative offset between hand and object**. It never trusts the pointmap's
absolute depth. [P §3.1, App. A]

For each frame:

1. `c_hand^H` is the centroid of the *visible* HaWoR hand. For each pixel in the hand mask,
   cast a ray and take its first hit on the HaWoR mesh, then average the hits.
2. `c_hand^M` averages the MoGe points at **those same pixels**. `c_obj^M` averages the MoGe
   points over the object mask.
3. Scale ratio: `k = z_hand^H / z_hand^M`. This is the ratio of the two depths for the same
   physical surface.
4. Object target: `obj_target = c_hand^H + k · (c_obj^M − c_hand^M)`. Take the hand→object offset
   from the pointmap, rescale it into hand units, and attach it to the HaWoR hand.
5. Keep SAM 3D's orientation fixed. Solve a **one-dimensional** least squares for the translation
   scale `s`: `s* = tᵀ(obj_target − c_mesh) / tᵀt`. This slides the object along its own
   viewing ray and changes nothing else.
6. Rotate everything into a gravity-aligned frame with GeoCalib.

**Two things the released `scripts/optimize_translation_scale.py` does that the paper doesn't
say** (read 2026-09-26) [S]:
- **The object's size comes from one frame.** `run_pipeline.sh:186-190` passes `--ref-frame "$n"`
  (the anchor frame). The script computes k on that frame, then multiplies the mesh scale **and**
  every frame's translation by it (`mesh_scale = args.scale * ref_k`, lines ~309-374). Only after
  that does it solve the per-frame s. So the object's metric size rests on a single frame's
  hand–pointmap ratio, and it never averages down.
- **The object centroid counts back-side vertices.** `c_rot = verts_rotated[mesh_in_mask].mean()`
  keeps every vertex whose projection lands inside the object mask, front and back. The pointmap
  side, `o_pm`, sees only the front. The paper says "visible mesh vertices". The script's own
  hand-side comment warns that "a 2D silhouette filter cannot" exclude back-facing geometry, which
  is why the hand uses ray casting.

In a synthetic toy scene (a small study exercise not included here), the second point pulls the object about 2 cm toward the camera (2.20 cm
error vs 0.95 cm with front-facing vertices). The size from anchor frame 45 is +2.0% off, with a
±1.3% spread across possible anchors. These are toy magnitudes [M, toy]. The real bias grows with
the object's thickness, and R2 can measure it.

**Why this works.** Monocular metric depth is mostly wrong by a *global* scale factor in each
frame. A ratio of two depths taken from the same pointmap cancels that factor. The hand acts as
a ruler of roughly known length that appears in every frame. [I, but it is the paper's evident
reasoning]

**Where it breaks.** The first two points below are stated in the paper's Limitations. The rest are [I]:
- **Contact vs occlusion** [P §5]. Along the viewing ray, the depth gap between hand and object
  is the least-constrained number in the whole system. A finger in front of the object and a
  finger touching it look the same in 2D.
- **"Semi-accurate metric depth" is assumed** [P §5]. Step 4 still needs the MoGe *offset* to be
  right up to scale, and its direction to be right as well.
- **The hand ruler is an average hand.** A MANO shape estimate from monocular video can be off
  by the hand's own size error, and every object distance inherits that error. The paper
  never measures the metric error on its own clips; it has no ground truth for them. [I]
- **`k` is computed separately for each frame**, so it jitters. The release includes
  `scripts/smooth_trajectory.py`, but it is **commented out** in `run_pipeline.sh:321-326`. [S]
  **Measured on the authors' own whisking reconstruction (2026-09-26, `r2/RESULTS.md`)** [M]:
  - k has CV 2.6%, with a median frame-to-frame jump of 1.4%.
  - The anchor frame's k sizes the object 2.0% above what the clip-mean k would give.
  - **89% of the object path's frame-to-frame jitter lies along the viewing ray.** It is
    median 3.3 cm per frame at a depth of about 1 m.
  - Our own reconstruction shows the same pattern (88%).
- **Camera motion.** HaWoR runs with `--static_camera` [S `run_pipeline.sh:233`]. 14 of 187
  playbook clips were lost to camera motion [P §4.5].

**Related:**
- DUSt3R *normalises scale away*, VGGT learns a
  normalisation from data, and neither gives metric output. MoGe-2 is the metric branch of the
  same family. This paper shows the practical answer: **don't solve absolute scale in the
  geometry model; import it from an object whose size you know.**
- An **ArUco** tag of known size is the same kind of ruler as the hand, but exact. For
  real → sim scene reconstruction, use
  the tag as the metric anchor instead of trusting any depth model's absolute scale. [I]

---

## 4. The new tracker: guided flow matching on SAM 3D

### 4.1 The mechanism

SAM 3D's sparse-structure generator is a flow-matching DiT with one latent **block for shape**
and **several for pose**. Sampling integrates `ẋ = v_θ(x_t, t, c)` from noise at t=0 to data at
t=1 along `x_t = (1−t)x₀ + t·x₁`, with 25 Euler steps in the release. [P Eq. 1] [S `run_pipeline.sh:285`]

At every Euler step the tracker takes the model's normal update and then **blends each block
toward a target interpolant built from the same initial noise**:

```
x_next[k] = x_t[k] + v[k]·dt                          # the model's own Euler step
z_ref     = (1 − t_next)·ε_k + t_next·target_k        # where the target "would be" at t_next
x_next[k] = (1 − α_k)·x_next[k] + α_k·z_ref           # pull toward it
```

The code is `track_object.py:806-818` [S]. The shape target is the anchor frame's shape. The
pose target is the **previous frame's pose**.

- With **α = 1** this is RePaint-style replacement: the block is overwritten with the target's
  noised version at every step, so the model generates the *other* blocks **conditioned on**
  it. The release runs shape at `--guidance_strength 1` [S `run_pipeline.sh:269`]. The shape is
  therefore fully clamped, and each frame is effectively a sample from `p(pose | shape, image)`.
- With **0 < α < 1** it is a soft pull. The pose is biased toward the previous frame, which
  gives temporal coherence, but the image can still move it.

This works because shape and pose share one joint latent in the generator [P §3.1]. SAM 3D
already learned `p(shape, pose | image)`. Clamping the shape turns it into a pose estimator for
that specific object, **without training anything**.

**Connection to flow matching:** this is inference-time guidance of the same kind of ODE
that a flow-matching model integrates. A two-dimensional toy flow is enough to see the α trade-off and why
replacement at α=1 is exact conditioning only when the target block is consistent with the
image. [I]

### 4.2 Adaptive pose guidance

A fixed α_p causes two problems. If it is too strong, the object stays frozen at last frame's pose
("over-rigidity"). If it is too weak, the pose flips between symmetric modes. [P §3.1]

The paper's fix, for each pair of consecutive frames [P App. A] [S `track_object.py:1675-1677, 1753-1761`]:
1. Sample 20 points in the object mask and track them with BootsTAPIR.
2. Keep the points that stay inside the next frame's mask. Fit a closed-form 2D rigid transform
   by SVD to get the in-plane rotation `Δθ`.
3. `α_p = max(0.1, 0.7 − 0.09·|Δθ|)`.

**The paper leaves out the unit, and the code shows it is degrees per frame** (`rotation_deg`)
[S `track_object.py:1672`]. So α_p goes from 0.7 for a still object down to its floor of 0.1 at
≈ 6.7°/frame, which is about 200°/s at 30 fps. The logic: when the object turns fast, last
frame's pose is a stale target, so the tracker should rely less on it. Table 5 shows adaptive
beats fixed on both benchmarks (DexYCB CD 0.66 vs 0.74) [P].

### 4.3 Choosing among 25 samples per frame

Each frame draws **N = 25** candidate poses with different seeds [S `run_pipeline.sh:273`].

- **The principled choice** would score each candidate by the model's log-density, using the
  instantaneous change-of-variables integral restricted to the pose block (App. A, Eq. 2). That
  costs N·T·(1+D_p) = 25·25·14 ≈ **8.7k forward+backward passes per frame**, about 100× the cost
  of generation. [P]
  **Eq. 2 as printed has the wrong sign.** It reads log p₀(x₀) **+** ∫₀¹ tr(∂v/∂x) dt, but for a
  flow integrated from noise at t = 0 to data at t = 1 the correct form is **−** ∫. The release
  gets it right: it integrates from t = 1 down to 0 with dt < 0 and accumulates `div · dt`
  [S Fast-SAM3D `pose_log_likelihood.py:55, 62, 82, 85`]. A prototype on a 2-D ring toy flow
  (2026-09-26) matched the true density with correlation 0.9955 at T = 25. With the printed sign
  the correlation was −0.16. 
- **What they do instead:** cluster the samples under
  `d = w_t‖Δt‖ + w_r·2·arccos|⟨q_i,q_j⟩|`, which adds the translation distance to the SO(3)
  geodesic angle. Drop clusters smaller than 3, then rank the survivors by **render-IoU against
  the SAM 3 mask** [P App. A]. Release values: `cluster_w_rot 1.5`, `cluster_dist_thresh 0.3`,
  `cluster_min_size 3` [S `run_pipeline.sh:276-278`].
- **Result (Table 5):** clustering matches log-likelihood (DexYCB F-5 0.71 vs 0.72; HOI4D equal)
  at about **30× less** cost. Picking a random sample hurts on HOI4D (F-5 0.62 vs 0.72) but not
  on DexYCB. [P]

The idea behind it: confident samples agree with each other, while estimator noise spreads
across SE(3). The consensus stands in for the likelihood.

### 4.4 In the released run but not in the paper

The released command switches on several things the paper never mentions or ablates
[S `run_pipeline.sh:262-286`]:

- `--post_optimize`: SAM 3D's own layout refinement, **ICP against the pointmap plus
  silhouette render-and-compare** (PyTorch3D), run **on every frame**
  [S `sam3d_objects/pipeline/layout_post_optimization_utils.py`].
- `--chain_poses --chain_on_diffusion`: the pose chain passes latents from frame to frame.
- **Fast-SAM3D** itself, with Taylor-cache acceleration of the DiT (`ss_cache_stride 3`) and
  `--torch_compile`.

**Measured in R2 (2026-09-26)** [M]: rerunning the released tracker on the same clip reproduces
the authors' object **position** to 0.95 cm (median), but not its **orientation**:
- The difference is 7.6° at the anchor frame and grows to a **median of 44° at 85–125 frames
  from it**. The rank correlation with distance from the anchor is 0.35.
- The poses are *chained*, so orientation error accumulates with distance from the anchor.
  Translation is re-anchored to the hand on every frame by the scale step.
- The release's whisking anchor is frame 125 of 138, which means 125 frames of backward
  chaining.
- Render-IoU does not flag the bad frames: its rank correlation with orientation disagreement
  is −0.03.

**Open question for the replication:** Table 5 ablates guidance and selection but **not
post-optimisation**. How much of the gain over FoundationPose in Table 2 comes from guided
diffusion, and how much from per-frame render-and-compare? This can be checked with one flag.
Its size is unknown. [I]

---

## 5. Retargeting: SPIDER plus three robustness additions

### 5.1 The base optimizer

The base is SPIDER, an annealed MPPI in the style of DIAL-MPC. The hyperparameters in the
release's `config/override/do_as_i_do.yaml` **match the paper's Table 4** [S] [P]:

- **Reference.** mink IK matches MANO fingertips to Sharpa fingertips, with several random
  restarts to avoid local minima [P App. B]. The hand is a **floating 6-DoF wrist** plus fingers.
  The UR3e arm is solved *afterwards* in `deployment/` [S].
- **Control parameterisation.** Knots every 0.2 s are interpolated over a **3.0 s horizon**. The
  simulator steps at 200 Hz (`sim_dt 0.005`), and the plan is redone every **0.5 s**.
- **Per replan.** Up to **32 iterations**. Each iteration samples **1024** control sequences as
  the mean plus noise.
- **Two annealing schedules.**
  - Across the horizon, a log-spaced ramp from 1.0 to 4.0, so later knots explore more.
  - Across iterations, `β^i` with `β = 0.01^(1/32) ≈ 0.866`. By iteration 31 the noise is about
    1% of iteration 0's. [S `config.py:495`, `sampling.py:338`]
- **Update.** A softmax over **only the top 10%** of samples, using z-normalised rewards at
  temperature 1.0 [S `sampling.py:210-216`]. The temperature is not in the paper's table.
- **Domain randomisation.** Worst case (the minimum) over the DR parameter sets [S `sampling.py:253`].
- **Early stop.** When the improvement is below 0.01 for 2 iterations in a row.
- **Reward.**
  - Weighted qpos error: object position 1.0, object rotation 0.3, wrist position 0.1, wrist
    rotation 0.03, joints 0.01.
  - Minus penetration² × 3000 beyond a 5 mm margin.
  - Minus the transition penalty.
  - The terminal reward is weighted × 10.

### 5.2 The three additions, as implemented

**Warmup** (3.0 s, set equal to the horizon) [P §3.2] [S `sampling.py:111-164`]. For the first
3 s the object is **welded** in place while the hand moves freely. During warmup the reward
counts at × 0.1 and there is no perturbation. Then the weld drops. The optimizer uses this time
to find a stable grasp *before* the object has to move. This addition does most of the work:
25% → 66% on in-the-wild references.

**Random force perturbation** [S `mjwp.py:1405-1475`]:
- Each of the 1024 candidates is rolled out in **K = 4 worlds**, and the reward is averaged, so
  every iteration runs **4096 parallel MuJoCo Warp worlds** [S `sampling.py:95`].
- The perturbations use **common random numbers**: every candidate sees the same K disturbance
  realisations, so candidates are compared under identical pushes.
- Force magnitude is uniform in **[0, 0.5·m·g]**, up to half the object's weight, in a random
  direction. Torque is scaled the same way.
- The perturbation is a sticky process: it starts with p = 0.05 per step and continues with
  p = 0.95, so a push lasts about 20 steps, or 0.1 s [I from the parameters].
- It is **gated to airborne held frames** only.

This is domain randomisation applied to a *planner* rather than to a policy: it rejects grasps
that balance on fingertips. It barely moves the success numbers (+1 point) but "noticeably
improves" grasp naturalness [P §4.3]. That is a qualitative claim, not a measurement.

**Transition penalty.** It is called `pedestal_penalty_scale: 0.5` in the code; the paper calls
it `transition_penalty_scale` [S `mjwp.py:992-1075`]. On each frame, the reference state
switches one branch on:
- The reference says **in hand and away from the surface**: penalise worlds with **no
  hand–object contact**.
- The reference says **resting on the surface and not in hand**: penalise worlds with **no
  object–surface contact**.
- Frames in the middle of a transition are left unpenalised.

The object starts on an **auto-placed pedestal** rather than the floor [S `launch.py:26-29`].
This matches the "object base plate" trick in App. B, which lets objects start in any pose, for
example a spoon upright in a cup.

### 5.3 Results (Table 3) [P]

Success means mean `E_pos < 0.1 m` **and** `E_rot < 0.5 rad`.

| | Reconstructed (655) | | | OakInk2 (1,352, MoCap) | | |
|---|---|---|---|---|---|---|
| | Success | Pos | Rot | Success | Pos | Rot |
| Annealed Sampling (SPIDER) | 0.25 | 0.08 | 0.40 | 0.72 | 0.08 | 0.32 |
| + Warmup | 0.66 | 0.06 | 0.28 | 0.77 | 0.06 | 0.25 |
| + Perturbation | 0.67 | 0.06 | 0.30 | 0.79 | 0.03 | 0.14 |
| + Transition | **0.71** | 0.05 | 0.28 | **0.81** | 0.03 | 0.15 |

### 5.3a The reported error averages in the warmup frames

Warmup prepends `warmup_steps` reference frames (600 here) that repeat frame 0 while the
object is welded [S `utils/io.py:101-119`]. The final error is the mean over **every**
simulated frame, including those [S `optimize_physics.py:747-752`]. On the whisking clip, 600
of 1600 frames are warmup, and their object error is about 0.1 cm by construction. So the logged
error is **about 36% lower than the error over the frames that actually track the video** [M].
The details are in `r1/RESULTS.md`.

Whether the paper's Table 3 used this same all-frames mean is not stated [I]. If it did, the
"+Warmup" rows are measured over a different set of frames than the "Annealed" row, which has
no warmup frames. On clips near the 10 cm / 0.5 rad threshold, that can turn a failure into a
success.

### 5.4 Compute, measured on the 5080 [M]

- One iteration runs 4096 worlds × 600 steps (3 s at 200 Hz), which is about 2.5 M world-steps.
  On the 5080 that takes **about 4 s**, at 96% GPU utilisation, **6.6 GB peak**. The 16 GB card is
  enough.
- **Early stopping, not the 32-iteration cap, sets the cost.** The shipped run stopped at the
  2-iteration minimum on 10 of its 16 replans and averaged 4.0 iterations. Our rerun averaged
  5.25. My earlier worst-case estimate (32 iterations every replan, about 1.6 B world-steps)
  overstated the real cost by about 8× [M].
- **One whisking clip (4.7 s of video plus 3 s of warmup) takes 8.5 min on the 5080**, plus a
  one-off Warp kernel compile of about 2.5 min on the first run. The paper says "minutes" [P §1],
  and that holds.

---

## 6. Results, read critically

- **The Table 2 gains on standard benchmarks are small.** DexYCB F-5 is 0.71 vs FoundationPose's
  0.69 and F-10 is 0.93 vs 0.89. HOI4D is 0.72 vs 0.71 and 0.91 vs 0.91. Chamfer distance on
  DexYCB is the clearest win (0.66 vs 0.89). The comparison itself is clean: FoundationPose and
  Any6D are plugged into the *same* pipeline with the same mesh, so only the tracker changes. [P]
- **The big claim is in-the-wild, and it is a preference study.**
  - Setup: 150 videos, 3 ratings per video from a pool of 5 raters, left/right order randomised.
  - Result: 67% prefer ours, 18% prefer FoundationPose, 15% are ties, which is 79% among
    non-ties. 75% of videos had unanimous ratings, and Fleiss' κ = 0.65. [P App. C]
  - What is not stated: who the raters were. The benchmark videos have not been released.
- **Real-robot evidence is qualitative.**
  - The rollouts are open-loop replays at "roughly half speed" [P App. D].
  - The initial x, y, z and yaw are **aligned by hand** to the workspace [P §4.4].
  - They show 10 tasks and report **no success rate**.
- **No policy is trained on the 500 generated trajectories.** The paper shows the data can be
  replayed. It does not show that the data is useful for *learning*. That leaves an obvious
  next step open for someone else.
- **The success threshold is generous.** A mean position error of 10 cm is loose for picking up
  a whisk. [I]
- **The 100DOH filtering funnel adds up** [P §4.5]:
  - 2000 clips → 187 with meaningful hand–object interaction.
  - 187 − 41 (out of frame) − 29 (no activity or across a shot boundary) − 14 (camera motion)
    − 10 (SAM 3D failure) − 10 (other) = **83 (4%)**.

## 7. What the release does not include

Replication has to build these parts itself:

| Missing | Needed for | Route |
|---|---|---|
| Eval harness for F-5, F-10 and Chamfer on DexYCB (160 clips) and HOI4D (12 clips), with GT hands | Tables 2 and 5 | write it; the datasets need registration and agreement |
| OakInk2 processing for the Do-as-I-Do fork | Table 3, right half | upstream SPIDER has `examples/config/override/oakinkv2.yaml` and `spider/process_datasets/`; port the three additions onto it, or port OakInk2 onto the fork |
| The 655 reconstructed references and the 150-video in-the-wild set | Table 3 left half, Fig. 5 | not released; build your own set |
| Robot models other than `sharpa` and `mano` | any other hand | upstream SPIDER ships ability, allegro, inspire, xhand, schunk and metahand; MuJoCo Menagerie has `leap_hand`, `sharpa_wave`, `wonik_allegro` and **`robotstudio_so101`** |
| Sharpa SDK, UR3e | Fig. 6 real deployment | proprietary hardware and SDK |

What the release *does* include is a **finished whisking demo**: the raw clip, the tracking
output, `scene.xml`, `trajectory_kinematic.npz` and `trajectory_mjwp.npz`. The retargeting half
can therefore be exercised and checked against a known output **before any reconstruction
runs**. [S]

Licences: HaWoR is **CC BY-NC-ND 4.0**, SAM 3 and SAM 3D use the SAM License, and MANO is
non-commercial. That is fine for research and a workshop paper, but not for anything commercial.

---

## 8. What the paper offers for robot data at scale

- **A complete data-generation pipeline with a measurable cost/yield curve:** GPU-minutes per
  clip, survival rate and retarget success can each be measured end to end (R1, R2 and R5 in
  `PLAN.md`).
- **A clear efficiency target.** The cost sits in specific places: 25 samples × 25 Euler steps
  per frame in the tracker, 4096 worlds per MPC iteration, Taylor caching, and clustering in
  place of likelihood. R6 maps that trade-off.
- **Sim-to-real in miniature.** Perturbation-robust planning in MuJoCo Warp, a digital-twin
  replay step and CoACD collision meshes make it real → sim → real at the object level.
- **An obvious extension.** The stated limitation is "hand and object only, **no scene**"
  [P §5]. Adding scene reconstruction would give *video → scene + object + hand → simulator*.
- **Methods worth knowing:** guided sampling of a flow-matching DiT, annealed MPPI, and a
  preference study designed around κ, ties and position randomisation.

**On absolute depth:** the paper *assumes* its metric scale is good enough and never measures
that on real clips (§3). R4 in `PLAN.md` measures it, with an ArUco tag of known size as ground
truth.
