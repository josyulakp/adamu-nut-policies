---
title: ADAM-U Nut Pick Eval
emoji: 🔩
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Place a nut, pick a policy, watch it pick and place
models:
- josyula/adamu-nut-policies
---

# ADAM-U nut pick-and-place: interactive evaluation

Drag the nut and the green target on the table map, choose one of 19 trained policies (Diffusion Policy, ACT,
DP + DSRL, Touch in the Wild, DexScrew) and a condition, and press **Pick**. The episode runs here in MuJoCo and
streams back: the robot in 3D, the camera image the policy saw and, for the tactile policies, the FlexiTac pads.

The models are in [`josyula/adamu-nut-policies`](https://huggingface.co/josyula/adamu-nut-policies), where the model
card has the results. This Space serves the page and the API that the GitHub Pages copy of the page calls:

| Endpoint | |
|---|---|
| `GET /api/config` | models, conditions, table regions, ideal success rates |
| `POST /api/preview {model, seed, layout?}` | initial scene for a layout |
| `POST /api/run {model, cond, seed, layout}` | queue an episode, returns `{job}` |
| `GET /api/job?id=..&from=k` | state, queue position, frames from `k`, result |

`layout` is `{"nut": [x, y, yaw], "target": [x, y]}` in metres and radians, world frame (robot waist at the
origin, +x forward, +y to the robot's left).

On the free CPU hardware an episode takes about 5 to 10 s for DP, ACT, DSRL and DexScrew and about a minute for Touch
in the Wild. Episodes queue: one runs at a time per Python environment. Code: the `adamu_nut_yam/`, `tw/`, `ds/`
folders here; the README of the full repository explains each method.
