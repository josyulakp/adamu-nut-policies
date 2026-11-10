#!/bin/bash
# Stage the Hugging Face Space folder (code + meshes + web assets, no models) and optionally upload it.
#   space/make_space.sh                 stage into ./space_build
#   space/make_space.sh --upload [id]   ... and upload to the Space (default josyula/adamu-nut-eval)
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd); OUT=$REPO/space_build; ID=${2:-josyula/adamu-nut-eval}
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$REPO"
[ -f webapp/static/scene_tac.json ] || (source adamu_nut_yam/cotrain/env.sh && python webapp/build_assets.py)
cp space/Dockerfile space/README.md "$OUT/"
cp -r setup adamu_nut_yam adamu_nut_tactile tw ds webapp assets "$OUT/"
rm -rf "$OUT/webapp/logs" "$OUT/adamu_nut_yam/cotrain/"{runs,data} "$OUT/tw/"{runs,data} "$OUT/ds/runs"
find "$OUT" -name __pycache__ -prune -exec rm -rf {} +
printf '.git\n' > "$OUT/.dockerignore"
du -sh "$OUT"
if [ "$1" = --upload ]; then
  hf repos create "$ID" --type space --sdk docker --public --exist-ok
  hf upload "$ID" "$OUT" . --repo-type space --commit-message "Update eval Space"
fi
