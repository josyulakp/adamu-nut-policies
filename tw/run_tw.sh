#!/bin/bash
# Touch in the Wild on the simulated FlexiTac nut task, end to end:
#   1. demonstrations -> UMI replay buffer (.zarr.zip)          tw/collect.py
#   2. visuo-tactile MAE pretraining                              their pretrain_mae
#   3. three diffusion policies with their UMI workspace:          their train.py
#        dp_vision (wrist camera only), dp_tactile_cnn (+ tactile SimpleCNN), dp_tactile_mae (+ pretrained MAE)
#   4. evaluation in MuJoCo, nominal and bump, 50 held-out seeds  tw/eval_policy.py
# Usage: tw/run_tw.sh   (OUT=<dir> for the large outputs: dataset ~1.5 GB, runs ~20 GB per policy with checkpoints)
set -e
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname $HERE)
TW=$REPO/ext/touch_in_the_wild
OUT=${OUT:-$HERE/runs}; EPOCHS=${EPOCHS:-25}; MAE_EPOCHS=${MAE_EPOCHS:-30}
source $REPO/.venv-tw/bin/activate
export WANDB_MODE=disabled MUJOCO_GL=egl PYTHONPATH=$TW TW_DATASET=${TW_DATASET:-$HERE/data/nut_tactile.zarr.zip}
mkdir -p $OUT $(dirname $TW_DATASET) $REPO/results/tw

# 1. 300 scripted demos (30 % start with a missed grasp, 20 % with the bump; the expert re-grasps), recorded through
#    release, retreat and a 10-step hold
[ -f $TW_DATASET ] || python $HERE/collect.py --n 300 --workers ${WORKERS:-24} --out $TW_DATASET > $OUT/collect.log 2>&1

cd $TW
# 2. MAE pretraining on the tactile images of the same dataset
if ! ls $OUT/mae/pretrain_checkpoints/*/epoch_$(printf %04d $((MAE_EPOCHS-1))).pth >/dev/null 2>&1; then
  mkdir -p $OUT/mae && (cd $OUT/mae && python -m pretrain_mae.pretrain_mae --config-dir $HERE/config task=sim_nut \
    training.num_epochs=$MAE_EPOCHS training.ema_decay=0.99 dataloader.num_workers=12 > $OUT/mae.log 2>&1)
fi
MAE=${MAE:-$(ls -t $OUT/mae/pretrain_checkpoints/*/epoch_*.pth | head -1)}

# 3. policies
train() {
  local name=$1; shift
  [ -f $OUT/$name/DONE ] && return
  ACCELERATE_MIXED_PRECISION=bf16 python train.py --config-name train_diffusion_unet_timm_umi_workspace --config-dir $HERE/config \
    task=sim_nut logging.mode=disabled hydra.run.dir=$OUT/$name training.num_epochs=$EPOCHS \
    training.checkpoint_every=5 checkpoint.topk.k=1 dataloader.num_workers=16 val_dataloader.num_workers=8 "$@" \
    > $OUT/$name.log 2>&1
  touch $OUT/$name/DONE
}
train dp_vision      policy.obs_encoder.use_tactile=false
train dp_tactile_cnn policy.obs_encoder.use_tactile=true policy.obs_encoder.tactile_model_choice=simple_cnn
train dp_tactile_mae policy.obs_encoder.use_tactile=true policy.obs_encoder.tactile_model_choice=pretrain \
  policy.obs_encoder.pretrain_ckpt_path=$MAE

# 4. evaluation (3 rollouts per condition recorded to rollouts/tw/)
for name in dp_vision dp_tactile_cnn dp_tactile_mae; do
  for cond in nominal bump; do
    [ -f $REPO/results/tw/${name}_${cond}.json ] || python $HERE/eval_policy.py --ckpt $OUT/$name/checkpoints/latest.ckpt \
      --n 50 --cond $cond --out $REPO/results/tw/${name}_${cond}.json --record 3 --record_dir $REPO/rollouts/tw \
      > $OUT/eval_${name}_${cond}.log 2>&1
    tail -1 $OUT/eval_${name}_${cond}.log
  done
done
