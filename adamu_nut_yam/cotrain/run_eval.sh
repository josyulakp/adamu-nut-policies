#!/bin/bash
# Evaluate every image policy (Diffusion Policy, ACT, DSRL) on the 50 held-out seeds (2_000_000+) in one domain.
#   adamu_nut_yam/cotrain/run_eval.sh ideal            # the released models in models/
#   adamu_nut_yam/cotrain/run_eval.sh real runs        # your own checkpoints in adamu_nut_yam/cotrain/runs/
# domain: real (the real-robot stand-in), sim (randomised training simulator) or ideal (nominal, nothing perturbed).
# Results: results/cotrain/<model>_<domain>.json (per-seed success); one summary line per model in the .log files.
set -e
DOM=${1:-ideal}
cd "$(dirname "$0")"; source env.sh; REPO=$(cd ../.. && pwd)
if [ -n "$2" ]; then M=$(cd "$2" && pwd); DP=$M; ACT=$M; DSRL=$M; else DP=$REPO/models/dp; ACT=$REPO/models/act; DSRL=$REPO/models/dsrl; fi
OUT=${RESULTS:-$REPO/results/cotrain}; mkdir -p $OUT
for c in sim_only co_r10_a025 co_r10_a05 co_r50_a05 real_r10 real_r50 real_r100; do
  [ -f $DP/$c.pt ] && { [ -f $OUT/${c}_$DOM.json ] || python dp.py eval --ckpt $DP/$c.pt --domain $DOM --episodes 50 \
    --json $OUT/${c}_$DOM.json > $OUT/${c}_$DOM.log 2>&1 & }
done
for c in act_sim_only act_real_r50 act_co_r10_a05 act_co_r50_a05; do
  [ -f $ACT/$c.pt ] && { [ -f $OUT/${c}_$DOM.json ] || python act.py eval --ckpt $ACT/$c.pt --domain $DOM --episodes 50 \
    --json $OUT/${c}_$DOM.json > $OUT/${c}_$DOM.log 2>&1 & }
done
wait
for r in dsrl_co_r50 dsrl_real_r50; do
  [ -f $DSRL/$r/agent.pt ] && { [ -f $OUT/${r}_$DOM.json ] || DSRL_DOMAIN=$DOM python eval_dsrl.py --run $DSRL/$r \
    --json $OUT/${r}_$DOM.json > $OUT/${r}_$DOM.log 2>&1; }
done
grep -h success_rate $OUT/*_$DOM.log
