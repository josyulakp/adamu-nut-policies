#!/usr/bin/env python3
"""Interactive evaluation server: place the nut and the target, pick a model and a condition, watch the episode.

  webapp/run.sh                      (or: .venv/bin/python webapp/server.py --port 8770)
  then open http://localhost:8770    (from another machine: ssh -L 8770:localhost:8770 <server>)

Two worker processes (webapp/worker.py) keep the models loaded on the GPU: 'cam' in .venv (Diffusion Policy, ACT,
DSRL) and 'tac' in .venv-tw (Touch in the Wild, DexScrew). Each runs one episode at a time; requests queue.
API
  GET  /api/config                    models, conditions, table regions, ideal success rates
  POST /api/preview {model, seed, layout?}          initial scene for a layout (or the seed's own layout)
  POST /api/run     {model, cond, seed, layout}     -> {job}
  GET  /api/job?id=..&from=k          {state, queue, status, start, frames[k:], end}
"""
import argparse
import os
import base64
import json
import subprocess
import sys
import threading
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "adamu_nut_yam"))
import registry as R  # noqa: E402

LAUNCH = {
    "cam": f"source '{REPO}/adamu_nut_yam/cotrain/env.sh' && exec python '{HERE}/worker.py' cam",
    "tac": (f"source '{REPO}/.venv-tw/bin/activate' && export MUJOCO_GL=${{MUJOCO_GL:-egl}} PYTHONPATH='{REPO}/ext/touch_in_the_wild' && "
            f"exec python '{HERE}/worker.py' tac"),
}
JOBS, LOCK = {}, threading.Lock()
ALLOW_ORIGIN = os.environ.get("ALLOW_ORIGIN", "*")     # who may call the API from a browser (the GitHub Pages site)
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", 8))        # episodes waiting per worker before new requests are refused


class Worker:
    def __init__(self, family, log):
        self.family, self.queue, self.ready = family, [], False
        self.p = subprocess.Popen(["bash", "-c", LAUNCH[family]], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(log, "a"), text=True, bufsize=1)
        threading.Thread(target=self.read, daemon=True).start()

    def submit(self, job, req):
        with LOCK:
            self.queue.append(job)
        self.p.stdin.write(json.dumps(dict(req, job=job)) + "\n"); self.p.stdin.flush()

    def read(self):
        for line in self.p.stdout:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("type") == "ready":
                self.ready = True; continue
            with LOCK:
                j = JOBS.get(d.get("job"))
                if j is None:
                    continue
                t = d["type"]
                if t == "status":
                    j["state"], j["status"] = "running", d["msg"]
                elif t == "start":
                    j["start"] = d
                elif t == "frame":
                    j["frames"].append({k: d[k] for k in ("t", "pose", "grip", "img", "tact", "xy_mm", "h_mm", "contact")})
                elif t == "end":
                    j["end"], j["state"] = d, "done"
                    if j["id"] in self.queue:
                        self.queue.remove(j["id"])
        with LOCK:                                   # worker died: fail its queued jobs
            for jid in self.queue:
                JOBS[jid].update(state="done", end={"success": False, "error": f"{self.family} worker exited"})


class Previewer:
    """Initial state for a layout, computed here (no policy needed): robot at its home pose, nut settled."""

    def __init__(self):
        import numpy as np
        from env import NutPickPlaceEnv
        self.np = np
        self.envs = {"yam": NutPickPlaceEnv(),
                     "tac": NutPickPlaceEnv(scene=REPO / "adamu_nut_tactile" / "assets" / "scene.xml")}
        self.lock = threading.Lock()

    def __call__(self, variant, seed, layout):
        np = self.np
        with self.lock:
            env = self.envs[variant]; env.layout = layout
            env.reset(seed)
            nut, tgt = env.nut_pos(), env.target_pos()
            P = np.concatenate([env.d.xpos, env.d.xquat], 1).astype(np.float32)
            return {"pose": base64.b64encode(P.tobytes()).decode(), "target": [float(x) for x in tgt],
                    "nut": [float(nut[0]), float(nut[1]), float(env.nut_yaw())]}


def ideal_rates():
    out = {}
    for mid, s in R.MODELS.items():
        cond = "ideal" if s["worker"] == "cam" else "nominal"
        f = R.DATA / "results" / f"{s['res']}_{cond}.json"
        if f.exists():
            d = json.loads(f.read_text()); e = d["episodes"] if isinstance(d, dict) else d
            out[mid] = [sum(bool(x["success"]) for x in e), len(e)]
    return out


def make_handler(workers, preview):
    class H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(HERE / "static"), **kw)

        def log_message(self, fmt, *args):
            if "/api/job" not in self.path:
                sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

        def send_json(self, obj, code=200):
            b = json.dumps(obj, separators=(",", ":")).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json"); self.cors()
            self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(b)

        def cors(self):
            self.send_header("Access-Control-Allow-Origin", ALLOW_ORIGIN)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def do_OPTIONS(self):
            self.send_response(204); self.cors(); self.send_header("Content-Length", "0"); self.end_headers()

        def body(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/api/config":
                return self.send_json({"models": list(R.MODELS.values()), "rates": ideal_rates(),
                                       "regions": {"nut": R.NUT_REGION, "target": R.TARGET_REGION, "place": R.PLACE_BOUNDS,
                                                   "table": R.TABLE},
                                       "workers": {k: w.ready for k, w in workers.items()}})
            if u.path == "/api/job":
                q = parse_qs(u.query); jid = q.get("id", [""])[0]; k = int(q.get("from", ["0"])[0])
                with LOCK:
                    j = JOBS.get(jid)
                    if j is None:
                        return self.send_json({"error": "unknown job"}, 404)
                    w = workers[j["worker"]]
                    pos = w.queue.index(jid) if jid in w.queue else 0
                    return self.send_json({"state": j["state"], "queue": pos, "status": j.get("status"), "start": j.get("start"),
                                           "frames": j["frames"][k:], "end": j.get("end")})
            return super().do_GET()

        def do_POST(self):
            u = urlparse(self.path)
            try:
                req = self.body()
                spec = R.MODELS[req["model"]]
            except (KeyError, ValueError) as e:
                return self.send_json({"error": f"bad request: {e}"}, 400)
            seed = int(req.get("seed", 2_000_000))
            layout = req.get("layout") or None
            if layout:
                (x0, x1), (y0, y1) = R.PLACE_BOUNDS
                for k in ("nut", "target"):
                    if k in layout:
                        v = [float(x) for x in layout[k]]
                        v[0], v[1] = min(max(v[0], x0), x1), min(max(v[1], y0), y1)
                        layout[k] = v
            if u.path == "/api/preview":
                return self.send_json(preview(spec["variant"], seed, layout))
            if u.path == "/api/run":
                cond = req.get("cond") or spec["conds"][0][0]
                if cond not in [c for c, _ in spec["conds"]]:
                    return self.send_json({"error": f"{spec['id']} does not support condition {cond}"}, 400)
                if len(workers[spec["worker"]].queue) >= MAX_QUEUE:
                    return self.send_json({"error": "busy: too many episodes queued, try again in a minute"}, 429)
                jid = uuid.uuid4().hex[:12]
                with LOCK:
                    JOBS[jid] = dict(id=jid, worker=spec["worker"], state="queued", frames=[], model=spec["id"])
                    if len(JOBS) > 200:                          # forget the oldest finished episodes
                        for old in [k for k, v in JOBS.items() if v["state"] == "done"][:50]:
                            del JOBS[old]
                workers[spec["worker"]].submit(jid, dict(model=spec["id"], cond=cond, seed=seed, layout=layout))
                return self.send_json({"job": jid})
            self.send_json({"error": "not found"}, 404)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8770)))
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    a = ap.parse_args()
    if not (HERE / "static" / "scene_tac.json").exists():
        subprocess.run([sys.executable, str(HERE / "build_assets.py")], check=True)
    logs = REPO / "webapp" / "logs"; logs.mkdir(exist_ok=True)
    workers = {f: Worker(f, logs / f"worker_{f}.log") for f in ("cam", "tac")}
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(workers, Previewer()))
    print(f"eval page: http://localhost:{a.port}  (workers warming up; logs in webapp/logs/)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
