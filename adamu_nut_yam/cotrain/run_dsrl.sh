#!/bin/bash
# DSRL-NA steering of frozen Diffusion Policies, online in the real stand-in (1500 episodes each, eval every 250).
# Needs runs/co_r50_a05.pt and runs/real_r50.pt from run_cotrain.sh (or copy models/dp/*.pt there).
cd "$(dirname "$0")"; source env.sh
python dsrl.py --base runs/co_r50_a05.pt --out runs/dsrl_co_r50 --episodes 1500 > runs/dsrl_co_r50.log 2>&1 &
python dsrl.py --base runs/real_r50.pt  --out runs/dsrl_real_r50 --episodes 1500 > runs/dsrl_real_r50.log 2>&1 &
wait
echo DSRL_DONE
