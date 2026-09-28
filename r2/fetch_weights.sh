#!/bin/bash
# R2 weights (pod). Equivalent to reconstruction/setup/02_fetch_weights.sh --download for the
# stages we run, minus HaWoR, with one deliberate difference:
#   02_fetch_weights.sh rsyncs the Hugging Face checkpoint *configs* into BOTH submodules'
#   checkpoints/hf/, overwriting same-named files. Fast-SAM3D tracks its own configs there in
#   git (an edited pipeline.yaml + slat_generator_faster.yaml), so that rsync would silently
#   replace the fork's pipeline config with upstream's. Here: sam-3d-objects (tracks none) gets
#   the HF configs; Fast-SAM3D keeps its tracked configs and only gains files it lacks (cp -n).
#   Heavy weights are stored once and symlinked into both, as the original does.
set -eo pipefail
python3 -m pip install -q 'huggingface-hub[cli]<1.0'
R=~/rig_work/do-as-i-do/reconstruction
W=$R/weights; SHARED=$W/sam3d_shared/hf
mkdir -p "$SHARED" "$W/tapnet"
t0=$(date +%s)
hf download facebook/sam-3d-objects --local-dir "$W/sam3d-download" --max-workers 8 > /dev/null
dl="$W/sam3d-download/checkpoints"
HEAVY="slat_generator.ckpt slat_encoder.ckpt ss_generator.ckpt ss_encoder.ckpt slat_decoder_mesh.ckpt slat_decoder_mesh.pt slat_decoder_gs.ckpt slat_decoder_gs_4.ckpt ss_decoder.ckpt ss_encoder.safetensors"
for f in $HEAVY; do [ -f "$dl/$f" ] && mv -n "$dl/$f" "$SHARED/$f"; done
for name in sam-3d-objects Fast-SAM3D; do
  dst="$R/modules/$name/checkpoints/hf"; mkdir -p "$dst"
  for f in "$dl"/*; do [ -f "$f" ] && cp -n "$f" "$dst/"; done      # never overwrite a tracked config
  for f in $HEAVY; do [ -f "$SHARED/$f" ] && ln -sf "$SHARED/$f" "$dst/$f"; done
done
( cd "$R/modules/Fast-SAM3D" && git status --short checkpoints/ | head -5; echo "Fast-SAM3D tracked configs unchanged: $(git diff --quiet -- checkpoints/ && echo yes || echo NO)" )
echo "sam3d weights $(du -shL "$SHARED" | cut -f1) in $(( $(date +%s) - t0 ))s"
wget -q -nc https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt -P "$W/tapnet"
hf download facebook/sam3 > /dev/null && echo "sam3 cached"
hf download Ruicheng/moge-vitl > /dev/null && echo "moge-vitl cached"
echo "weights done in $(( $(date +%s) - t0 ))s"; df -h / | tail -1
