#!/bin/bash
# Co-training experiment: 7 Diffusion Policies (1 training seed each), evaluated on 50 held-out real-domain episodes.
cd "$(dirname "$0")"; source env.sh; mkdir -p runs data
# demonstrations (successful scripted episodes only): 300 in sim, 100 in the real-robot stand-in
[ -f data/sim.npz ]  || python collect.py --domain sim  --n 300 --out data/sim.npz  > runs/collect_sim.log 2>&1
[ -f data/real.npz ] || python collect.py --domain real --n 100 --out data/real.npz > runs/collect_real.log 2>&1
STEPS=${STEPS:-20000}
train() { [ -f runs/$1.pt ] || python dp.py train --steps $STEPS --out runs/$1.pt "${@:2}" > runs/$1.log 2>&1; }
ev() { [ -f runs/$1_$2.json ] || python dp.py eval --ckpt runs/$1.pt --domain $2 --episodes 50 --json runs/$1_$2.json > runs/$1_$2.log 2>&1; }
# wave 1: runs that hold the sim set on the GPU
train sim_only      --alpha 0    --n_sim 300 &
train co_r10_a025   --alpha 0.25 --n_sim 300 --n_real 10 &
train co_r10_a05    --alpha 0.5  --n_sim 300 --n_real 10 &
train co_r50_a05    --alpha 0.5  --n_sim 300 --n_real 50 &
wait
# wave 2: real-only baselines; evaluate wave 1 meanwhile
train real_r10      --alpha 1 --n_real 10 &
train real_r50      --alpha 1 --n_real 50 &
train real_r100     --alpha 1 --n_real 100 &
for r in sim_only co_r10_a025 co_r10_a05 co_r50_a05; do ev $r real & done
ev sim_only sim & ev co_r50_a05 sim &
wait
for r in real_r10 real_r50 real_r100; do ev $r real & done
wait
echo COTRAIN_DONE
