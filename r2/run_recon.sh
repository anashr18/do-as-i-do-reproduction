#!/bin/bash
# R2 driver: reconstruction/run_pipeline.sh at 824591b, stage for stage and flag for flag,
# with four recorded deviations:
#   D1  ffmpeg: `-vsync 0` -> `-fps_mode passthrough`. The sam3 env's conda-forge ffmpeg (9.0.2)
#       no longer accepts -vsync, so run_pipeline.sh's stages 0 and 1 fail as written.
#   D2  SAM 3 object prompt: two positive points on the whisk in frame 125 (the metal collar
#       and the wire loops), in place of the interactive --click, which needs a display.
#   D3  HaWoR not run (no MANO licence). The authors' shipped HaWoR output
#       (whisking/whisking/all_hand_meshes.npz) is copied in where HaWoR would write it.
#   D4  Runs in a fresh directory with a copy of the clip, so the authors' shipped outputs
#       stay untouched as the comparison reference.
# Every stage logs wall time and peak GPU memory (nvidia-smi, 200 ms) to $RUN/stage_metrics.tsv.
set -eo pipefail
R=~/rig_work/do-as-i-do/reconstruction
source "$R/config/paths.sh"
source ~/miniforge3/etc/profile.d/conda.sh
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda

RUN=~/rig_work/r2_run; VIDEO_DIR=$RUN/whisking; mkdir -p "$VIDEO_DIR/whisking"
cp -n "$R/whisking/whisking.mp4" "$VIDEO_DIR/"
cp -n "$R/whisking/whisking/all_hand_meshes.npz" "$VIDEO_DIR/whisking/"          # D3
VIDEO_PATH=$VIDEO_DIR/whisking.mp4; VIDEO_NAME=whisking
n=125; OBJ_ID=whisk; ANCHOR_HAND=right
OBJ_POINTS="566,486;582,525"; OBJ_LABELS="1;1"                                      # D2
FRAME_PATH="$VIDEO_DIR/$(printf "%04d.png" "$n")"
POINTMAP_PATH="$VIDEO_DIR/$(printf "%04d_pointmap.npy" "$n")"
MASKS_DIR="$VIDEO_DIR/video_segmentation/masks/frame_$(printf "%06d" "$n")_masks"
VIDEO_MASKS_DIR="$VIDEO_DIR/video_segmentation/masks"
M=$RUN/stage_metrics.tsv; [ -f "$M" ] || echo -e "stage\twall_s\tgpu_peak_mib\texit" > "$M"

stage() {   # stage NAME CMD...  (skips a stage already logged ok, so the script can resume)
  local name=$1; shift
  if grep -q "^$name	.*	0$" "$M"; then echo "=== $name: done earlier, skipping"; return 0; fi
  echo "=== $name $(date +%T)"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > "$RUN/mem_$name.txt" &
  local mon=$! t0=$(date +%s%N) rc=0
  "$@" || rc=$?
  kill $mon 2>/dev/null; wait $mon 2>/dev/null || true
  local peak=$(sort -n "$RUN/mem_$name.txt" | tail -1) ds=$(( ($(date +%s%N) - t0) / 100000000 ))
  echo -e "$name\t$((ds / 10)).$((ds % 10))\t$peak\t$rc" | tee -a "$M"
  return $rc
}

conda activate "$ENV_SAM3"
mkdir -p "$VIDEO_DIR/all_frames"
stage s0_frames ffmpeg -v error -y -i "$VIDEO_PATH" -fps_mode passthrough -start_number 0 "$VIDEO_DIR/all_frames/%06d.png"   # D1
cat > "$VIDEO_DIR/config.json" <<EOF
{
    "frame_number": $n,
    "object_names": ["$OBJ_ID"],
    "anchor_hand": "$ANCHOR_HAND"
}
EOF
ffmpeg -v error -y -i "$VIDEO_PATH" -vf "select=eq(n\,${n})" -fps_mode passthrough -frames:v 1 "$FRAME_PATH"   # D1
cd "$SCRIPTS_DIR"
stage s1_sam3_object python run_sam3_video.py --video "$VIDEO_PATH" --points "$OBJ_POINTS" --point_labels "$OBJ_LABELS" --obj_id "$OBJ_ID" --frame_idx "$n"
stage s1_sam3_hand python run_sam3_video.py --video "$VIDEO_PATH" --text "$ANCHOR_HAND hand" --obj_id "${ANCHOR_HAND}_hand_0" --frame_idx "$n"

conda activate "$ENV_SAM3D"
cd "$SAM3D_DIR"
stage s2_sam3d_mesh python generate_mesh_sam3d.py --image_path "$FRAME_PATH" --masks_dir "$MASKS_DIR"
cd "$SCRIPTS_DIR"
stage s2_moge_ref python get_pointmap_dir.py --image "$FRAME_PATH" --output "$POINTMAP_PATH"
echo "=== s2_hawor: skipped (D3), using the shipped all_hand_meshes.npz"
stage s2_moge_all python get_pointmap_dir.py --image_dir "$VIDEO_DIR/all_frames"
stage s2_gravity python predict_video_gravity.py "$VIDEO_DIR/all_frames" --output_path "$VIDEO_DIR/gravity.json"

conda activate "$ENV_TAPNET"
cd "$SCRIPTS_DIR"
stage s25_tapir python tapir_velocity_tracking.py --video "$VIDEO_PATH" --mask-dir "$VIDEO_MASKS_DIR" --object "$OBJ_ID" --checkpoint "$TAPNET_CKPT"

conda activate "$ENV_SAM3D"
cd "$FASTSAM3D_DIR"
stage s3_track python track_object.py \
    --config checkpoints/hf/pipeline.yaml --vid_dir "$VIDEO_DIR" --masks_root "$VIDEO_MASKS_DIR" \
    --object_name "$OBJ_ID" --init_frame "$n" --output_dir "$VIDEO_DIR/obj_tracking_out/$OBJ_ID" \
    --guidance_strength 1 --save_layout --fix_scale_to_init_frame --pose_guidance_strength 0.5 \
    --num_pose_samples 25 --scoring_metric render_iou --pose_selection cluster \
    --cluster_dist_thresh 0.3 --cluster_min_size 3 --cluster_w_rot 1.5 --chain_poses --post_optimize \
    --no-enable_shape_icp --chain_on_diffusion --enable_ss_cache --torch_compile --euler_steps 25 \
    --rotvel_json "$VIDEO_DIR/perframe_tracking_$OBJ_ID/motion_stats.json"
cd "$SCRIPTS_DIR"
CV="$VIDEO_DIR/obj_tracking_out/$OBJ_ID/combined_visualization"
stage s3_project python run_project_mesh_combined.py --video "$VIDEO_PATH" --mesh "$MASKS_DIR/$OBJ_ID/${OBJ_ID}.obj" \
    --json "$CV/layout.json" --output-base "$CV/projected"
stage s3_camframe python convert_layout_to_camera_frame.py --input "$CV/layout.json" --output "$CV/layout_camera_frame.json"
stage s4_scale python optimize_translation_scale.py --video-dir "$VIDEO_DIR" --layout-json "$CV/layout_camera_frame.json" \
    --anchor-hand "$ANCHOR_HAND" --ref-frame "$n"
echo "=== R2 reconstruction complete $(date +%T)"
cat "$M"
