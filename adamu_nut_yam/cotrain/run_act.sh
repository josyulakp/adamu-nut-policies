#!/bin/bash
# ACT under the same co-training conditions as the Diffusion Policy runs (needs data/ from run_cotrain.sh).
cd "$(dirname "$0")"; source env.sh; mkdir -p runs
STEPS=${STEPS:-20000}
train() { [ -f runs/$1.pt ] || python act.py train --steps $STEPS --out runs/$1.pt "${@:2}" > runs/$1.log 2>&1; }
ev() { [ -f runs/$1_$2.json ] || python act.py eval --ckpt runs/$1.pt --domain $2 --episodes 50 --json runs/$1_$2.json > runs/$1_$2.log 2>&1; }
train act_sim_only    --alpha 0   --n_sim 300 &
train act_real_r50    --alpha 1   --n_real 50 &
train act_co_r10_a05  --alpha 0.5 --n_sim 300 --n_real 10 &
train act_co_r50_a05  --alpha 0.5 --n_sim 300 --n_real 50 &
wait
for r in act_sim_only act_real_r50 act_co_r10_a05 act_co_r50_a05; do ev $r real & done
ev act_sim_only sim &
wait
echo ACT_DONE
