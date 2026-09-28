"""R2: compare our reconstruction of the whisking clip with the authors' shipped one.

Runs on the pod in the sam3d env (numpy, scipy, trimesh).
    python compare.py [OURS_VIDEO_DIR] [SHIPPED_VIDEO_DIR]  ->  prints + writes r2_compare.json

Poses are compared the way retargeting consumes them (process_dataset.py:507-508, 709-713):
object vertices in camera frame = R(quat_wxyz_camera_frame) @ (mesh_scale * v) + translation_camera_frame.
"""

import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HOME = Path.home() / "rig_work"
OURS = Path(sys.argv[1]) if len(sys.argv) > 1 else HOME / "r2_run" / "whisking"
SHIP = Path(sys.argv[2]) if len(sys.argv) > 2 else HOME / "do-as-i-do" / "reconstruction" / "whisking"
OBJ, REF = "whisk", 125
LAYOUT = f"obj_tracking_out/{OBJ}/combined_visualization/layout_camera_frame_optimized.json"
MESH = f"video_segmentation/masks/frame_{REF:06d}_masks/{OBJ}/{OBJ}.obj"


def load(d: Path) -> dict:
    L = json.load(open(d / LAYOUT))
    tso = L["translation_scale_optimization"]
    fr = {o["frame_idx"]: o["local_to_scene"] for o in L["objects"]}
    return dict(
        frames=fr, mesh_scale=tso["mesh_scale"], mesh_scale_orig=tso["mesh_scale_original"],
        ref_k=tso["ref_frame_k"], k=np.array([p["pointmap_scale"] for p in tso["per_frame"]]),
        mesh=trimesh.load(d / MESH, force="mesh"),
    )


def jitter(t: np.ndarray) -> dict:
    d2 = np.diff(t, n=2, axis=0)
    a = np.linalg.norm(d2, axis=1)
    return dict(d2_median_cm=100 * float(np.median(a)), d2_p90_cm=100 * float(np.percentile(a, 90)),
                depth_share=float(np.median(np.abs(d2[:, 2]) / np.maximum(a, 1e-9))))


def world_points(m: dict, fi: int, n: int = 20000) -> np.ndarray:
    ls = m["frames"][fi]
    R = Rotation.from_quat(np.array(ls["quat_wxyz_camera_frame"])[[1, 2, 3, 0]]).as_matrix()
    p = m["mesh"].sample(n, return_index=False) if hasattr(m["mesh"], "sample") else m["mesh"].vertices
    return (R @ (m["mesh_scale"] * np.asarray(p)).T).T + np.array(ls["translation_camera_frame"])


def chamfer_cm(a: np.ndarray, b: np.ndarray) -> float:
    return 100 * 0.5 * (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean())


o, s = load(OURS), load(SHIP)
common = sorted(set(o["frames"]) & set(s["frames"]))
to = np.array([o["frames"][f]["translation_camera_frame"] for f in common])
ts = np.array([s["frames"][f]["translation_camera_frame"] for f in common])
qo = np.array([o["frames"][f]["quat_wxyz_camera_frame"] for f in common])[:, [1, 2, 3, 0]]
qs = np.array([s["frames"][f]["quat_wxyz_camera_frame"] for f in common])[:, [1, 2, 3, 0]]
dt = to - ts
ang = np.degrees((Rotation.from_quat(qo).inv() * Rotation.from_quat(qs)).magnitude())

ext = lambda m: (100 * m["mesh_scale"] * m["mesh"].bounding_box.extents).round(2).tolist()
out = dict(
    frames=dict(ours=len(o["frames"]), shipped=len(s["frames"]), common=len(common)),
    mesh=dict(vertices=dict(ours=len(o["mesh"].vertices), shipped=len(s["mesh"].vertices)),
              extent_cm=dict(ours=ext(o), shipped=ext(s)),
              mesh_scale=dict(ours=o["mesh_scale"], shipped=s["mesh_scale"]),
              mesh_scale_original=dict(ours=o["mesh_scale_orig"], shipped=s["mesh_scale_orig"]),
              chamfer_at_ref_frame_cm=chamfer_cm(world_points(o, REF), world_points(s, REF))),
    ref_k=dict(ours=o["ref_k"], shipped=s["ref_k"]),
    k=dict(ours=dict(mean=float(o["k"].mean()), cv_pct=float(100 * o["k"].std() / o["k"].mean())),
           shipped=dict(mean=float(s["k"].mean()), cv_pct=float(100 * s["k"].std() / s["k"].mean()))),
    translation_diff_cm=dict(
        norm_median=100 * float(np.median(np.linalg.norm(dt, axis=1))),
        norm_p90=100 * float(np.percentile(np.linalg.norm(dt, axis=1), 90)),
        norm_max=100 * float(np.linalg.norm(dt, axis=1).max()),
        depth_z_median_abs=100 * float(np.median(np.abs(dt[:, 2]))),
        lateral_xy_median=100 * float(np.median(np.linalg.norm(dt[:, :2], axis=1))),
        depth_z_mean_signed=100 * float(dt[:, 2].mean())),
    rotation_diff_deg=dict(median=float(np.median(ang)), p90=float(np.percentile(ang, 90)), max=float(ang.max())),
    chamfer_per_frame_cm=dict(zip(["median", "p90", "max"], np.percentile(
        [chamfer_cm(world_points(o, f, 4000), world_points(s, f, 4000)) for f in common], [50, 90, 100]).round(2).tolist())),
    jitter=dict(ours=jitter(to), shipped=jitter(ts)),
)
print(json.dumps(out, indent=1))
json.dump(out, open(HOME / "r2_compare.json", "w"), indent=1)
