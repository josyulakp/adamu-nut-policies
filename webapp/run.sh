#!/bin/bash
# Start the interactive eval page on this machine: webapp/run.sh [port]   (default 8770, listens on localhost only)
# From your laptop: ssh -L 8770:localhost:8770 <user>@<server>, then open http://localhost:8770
REPO=$(cd "$(dirname "$0")/.." && pwd)
source "$REPO/adamu_nut_yam/cotrain/env.sh"
exec python "$REPO/webapp/server.py" --port "${1:-${PORT:-8770}}"
