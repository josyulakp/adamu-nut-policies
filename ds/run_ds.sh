#!/bin/bash
# DexScrew on the simulated FlexiTac nut task, end to end:
#   stage 1  privileged teacher   ds/bc_teacher.py  (imitation of the scripted expert with DART noise; see README for
#                                                   why not PPO: ds/train_ds.py teacher is kept but did not converge)
#   stage 2  student              ds/train_ds.py student  (their ProprioAdapt: proprio + touch history -> teacher latent)
#   stage 3  tactile BC fine-tune ds/stage3_bc.py  (success-filtered rollouts in the shifted-physics 'real' stand-in)
#   evaluation of all three: nominal, bump, shifted; 50 held-out seeds each, 3 rollouts per condition recorded
# Usage: ds/run_ds.sh   (OUT=<dir> for training outputs, default ds/runs)
set -e
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname $HERE)
OUT=${OUT:-$HERE/runs}; STUDENT_STEPS=${STUDENT_STEPS:-10e6}
source $REPO/.venv-tw/bin/activate
export MUJOCO_GL=egl
cd $HERE; mkdir -p $OUT $REPO/results/ds $REPO/rollouts/ds
[ -f $OUT/teacher/bc_final.pth ] || python bc_teacher.py --out $OUT/teacher > $OUT/teacher.log 2>&1
[ -f $OUT/student/stage2_nn/model_last.ckpt ] || python train_ds.py student --teacher $OUT/teacher/bc_final.pth \
   --steps $STUDENT_STEPS --out $OUT/student > $OUT/student.log 2>&1
[ -f $OUT/stage3/model_ft.ckpt ] || python stage3_bc.py --student $OUT/student/stage2_nn/model_last.ckpt --out $OUT/stage3 \
   > $OUT/stage3.log 2>&1
for spec in "teacher teacher $OUT/teacher/bc_final.pth" "student student $OUT/student/stage2_nn/model_last.ckpt" \
            "ft student $OUT/stage3/model_ft.ckpt"; do
  set -- $spec
  for cond in nominal bump shifted; do
    [ -f $REPO/results/ds/${1}_${cond}.json ] || python eval_ds.py --ckpt $3 --kind $2 --cond $cond --n 50 \
      --out $REPO/results/ds/${1}_${cond}.json --record 3 --record_dir $REPO/rollouts/ds --tag $1 2>/dev/null | grep success_rate
  done
done
