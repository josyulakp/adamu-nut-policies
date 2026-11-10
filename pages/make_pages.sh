#!/bin/bash
# Stage the static GitHub Pages copy of the eval page. It talks to the model server on the Hugging Face Space.
#   pages/make_pages.sh [out_dir] [space_url]
#   defaults: ./pages_build, https://josyula-adamu-nut-eval.hf.space
# Then push out_dir to a GitHub repository and enable Pages (Settings > Pages > Deploy from branch: main, / root).
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-$REPO/pages_build}; API=${2:-https://josyula-adamu-nut-eval.hf.space}
SPACE_PAGE=https://huggingface.co/spaces/josyula/adamu-nut-eval; MODELS=https://huggingface.co/josyula/adamu-nut-policies
mkdir -p "$OUT"
cp "$REPO"/webapp/static/{scene_yam.json,scene_tac.json,meshes_yam.bin,meshes_tac.bin} "$OUT/"
sed -e "s|<meta name=\"eval-api\" content=\"\">|<meta name=\"eval-api\" content=\"$API\">|" \
    -e "s|  training; placements outside them test generalisation.</p>|  training; placements outside them test generalisation.</p>\n  <p class=\"sub\"><a href=\"$MODELS\">Trained models</a> · <a href=\"$SPACE_PAGE\">model server (Hugging Face Space)</a> (free CPU, so the first request after a quiet spell wakes it up, which takes a few minutes).</p>|" \
    "$REPO/webapp/static/index.html" > "$OUT/index.html"
grep -q "content=\"$API\"" "$OUT/index.html"
touch "$OUT/.nojekyll"
cp "$REPO/pages/README.md" "$OUT/README.md"
echo "staged $OUT (API $API)"
