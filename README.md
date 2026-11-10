# ADAM-U nut pick-and-place: five policy-learning methods in MuJoCo

The right arm of a PNDbotics ADAM-U, fitted with an i2rt YAM parallel-jaw gripper, picks a 22 mm hex nut lying flat
on a table and sets it upright on a green target pad. The nut's position and yaw and the pad's position are random
in every episode. This repository has the code, trained models, evaluation results and recorded rollouts for five
methods:

| Method | Paper | What the policy sees | Code |
|---|---|---|---|
| **Diffusion Policy (DP)** | Chi et al., RSS 2023; sim + real co-training after Maddukuri et al. 2025 | head + wrist RGB, joint angles | `adamu_nut_yam/cotrain/dp.py` |
| **ACT** | Zhao et al., RSS 2023 | head + wrist RGB, joint angles | `adamu_nut_yam/cotrain/act.py` |
| **DP + DSRL** | Wagenmaker et al., CoRL 2025 | same as DP (DP frozen; RL picks its noise) | `adamu_nut_yam/cotrain/dsrl.py` |
| **Touch in the Wild** | Zhu et al. 2025, [repo](https://github.com/XinyueZhuXY/touch_in_the_wild) | wrist RGB + FlexiTac touch, gripper pose | `tw/` + their code |
| **DexScrew** | Hsieh et al. 2025, [repo](https://github.com/x-robotics-lab/dexscrew) | proprioception + touch history | `ds/` + their code |

DP, ACT and DSRL use the plain YAM gripper. Touch in the Wild and DexScrew use the same gripper with two simulated
FlexiTac tactile pads (12 x 32 taxels each, `adamu_nut_tactile/tactile.py`).

## Results

Success rate on the same 50 held-out seeds (2,000,000 to 2,000,049, never used in training). The success test is
the same for all methods: after the gripper lets go, the nut is within 2 cm of the target, upright (tilt < 15°),
resting on the table and not touching the gripper, held for 10 consecutive control steps (0.5 s) within 300 steps
(15 s). With 50 episodes the 95% interval is about ±12 points.

| Method / model | Trained on | Ideal | Real stand-in | Bump | Shifted physics |
|---|---|---|---|---|---|
| DP, sim only (`models/dp/sim_only.pt`) | 300 sim demos | **80%** | 2% | | |
| DP, real only (`models/dp/real_r100.pt`) | 100 real-stand-in demos | 0% | **74%** | | |
| DP, real only (`real_r50.pt` / `real_r10.pt`) | 50 / 10 real demos | 0% / 0% | 44% / 8% | | |
| DP, co-trained (`models/dp/co_r50_a05.pt`) | 300 sim + 50 real, α 0.5 | 64% | **74%** | | |
| DP, co-trained (`co_r10_a05.pt` / `co_r10_a025.pt`) | 300 sim + 10 real, α 0.5 / 0.25 | 34% / 44% | 12% / 18% | | |
| ACT, sim only / real only (`models/act/`) | 300 sim / 50 real | 14% / 14% | 8% / 26% | | |
| ACT, co-trained (`act_co_r10_a05.pt` / `act_co_r50_a05.pt`) | 300 sim + 10 / 50 real | 14% / 18% | 10% / 42% | | |
| DP + DSRL (`models/dsrl/dsrl_co_r50`) | co-trained DP + 1500 online episodes in the stand-in | 72% | 64% | | |
| DP + DSRL (`models/dsrl/dsrl_real_r50`) | real-only DP + 1500 online episodes | 0% | 40% | | |
| Touch in the Wild, wrist camera only (`models/tw/dp_vision.pt`) | 300 sim demos | **76%** | | **58%** | |
| Touch in the Wild, camera + touch, SimpleCNN (`dp_tactile_cnn.pt`) | 300 sim demos | 66% | | 32% | |
| Touch in the Wild, camera + touch, MAE (`dp_tactile_mae.pt`) | 300 sim demos + visuo-tactile MAE | 66% | | **58%** | |
| DexScrew stage 1 teacher (`models/ds/teacher.pth`) | imitation of the scripted expert, randomised physics | **100%** | | 100% | 18% |
| DexScrew stage 2 student (`models/ds/student.ckpt`) | distilled from the teacher | **80%** | | 0% | 20% |
| DexScrew stage 3 fine-tune (`models/ds/stage3_ft.ckpt`) | student + BC on shifted-physics successes | 50% | | 0% | **58%** |

Conditions:

* **Ideal**: the nominal simulator with nothing randomised or perturbed (default table, lights and camera
  calibration, no image or encoder noise, no latency, nominal masses and friction).
* **Real stand-in** (DP / ACT / DSRL): a fixed second domain that plays the physical robot. It has a table material
  and warm lighting the sim randomisation never produces, a head camera 2 cm / 2° off calibration, blur and noise,
  150 ms latency, softer joint servos, 92% gravity compensation, extra joint friction, pad friction 0.7 and a 36 g
  nut. Its demos are "teleoperated": multimodal (either reachable pair of flats) and at varying speed.
* **Bump** (Touch in the Wild / DexScrew): as the gripper first closes on the nut, the nut is pushed 3 cm sideways,
  so the first grasp misses and the policy has to notice, re-open and re-grasp.
* **Shifted physics** (DexScrew, the stand-in for the real robot in its stage 3): nut 40 g instead of 30 g, nut
  friction 0.9 instead of 1.5, pad friction 0.7 instead of 1.0, arm gains 80%, 2 control steps of latency.

Blank cells were not evaluated. The camera-policy and tactile-policy groups use different grippers (plain vs
FlexiTac), so compare within a group. Per-seed results are in `results/`.

## Layout

```
setup/            setup.sh, pinned requirements for the two Python environments
assets/meshes/    robot, gripper and nut meshes used by the MuJoCo scenes
adamu_nut_yam/    the task: env.py (environment + scripted expert), kin.py (arm IK), assets/ (scenes)
  cotrain/        DP, ACT, DSRL: cam_env.py (camera env, domains), collect.py, dp.py, act.py, dsrl.py,
                  eval_dsrl.py, record.py, run_cotrain.sh, run_act.sh, run_dsrl.sh, run_eval.sh
adamu_nut_tactile/ tactile.py (FlexiTac pad model), assets/ (scenes with the tactile fingers)
tw/               Touch in the Wild: sim.py (UMI-convention env + experts), collect.py, eval_policy.py,
                  config/task/sim_nut.yaml, run_tw.sh
ds/               DexScrew: nut_rl_env.py (RL env), vec_env.py, heuristic.py (scripted expert),
                  bc_teacher.py, train_ds.py, stage3_bc.py, eval_ds.py, run_ds.sh
webapp/           interactive evaluation page: server.py, worker.py, registry.py, build_assets.py, static/index.html
models/           trained models (below)
rollouts/         recorded evaluation episodes (below)
results/          per-seed evaluation results (JSON) and training logs
ext/              upstream repos, cloned by setup.sh (not part of this repository)
```

## Setup

Linux with an NVIDIA GPU (all runs here used one 48 GB L40S and 32 CPU cores), Python 3.11 and
[uv](https://docs.astral.sh/uv/).

```bash
setup/setup.sh
```

This creates two environments, because the upstream Touch in the Wild code needs an older PyTorch:

* `.venv`: Diffusion Policy, ACT and DSRL (PyTorch 2.15, CUDA 13.2 wheels, MuJoCo 4.0).
* `.venv-tw`: Touch in the Wild and DexScrew (PyTorch 2.4.1, CUDA 12.1, MuJoCo 4.0).

It also clones Touch in the Wild and DexScrew into `ext/` at the commits used here. Both are used unmodified.
Rendering is headless (`MUJOCO_GL=egl`, set by the scripts). The first Touch in the Wild run downloads the CLIP
ViT-B/16 weights through timm.

The trained models and recorded rollouts are too large for GitHub and are on the Hugging Face Hub at
[`josyula/adamu-nut-policies`](https://huggingface.co/josyula/adamu-nut-policies) (3.8 GB). Download them into the
repository root, where `models/` and `rollouts/` are expected:

```bash
pip install -U huggingface_hub     # provides the hf command
hf download josyula/adamu-nut-policies --include "models/*" --include "rollouts/*" --local-dir .
```

---

## 1. Diffusion Policy (DP)

Image Diffusion Policy. One ResNet-18 per camera (GroupNorm, trained from scratch) encodes head and wrist RGB
(112 px, random 104 px crops) and joint angles for the last 2 steps. A 1-D temporal U-Net with FiLM conditioning
predicts 16 joint-target actions, of which 8 are executed before re-planning. It trains with 100-step DDPM and
samples with 16-step DDIM. With **co-training**, every batch draws a fraction α from the real-stand-in demos and the
rest from sim (α = 0: sim only; α = 1: real only).

All commands run from `adamu_nut_yam/cotrain` with `source env.sh`.

**Evaluate a released model**

```bash
cd adamu_nut_yam/cotrain && source env.sh
python dp.py eval --ckpt ../../models/dp/co_r50_a05.pt --domain ideal --episodes 50 --json /tmp/dp_co_ideal.json
```

`--domain` is `ideal`, `real` (the real stand-in) or `sim` (the randomised training simulator). To evaluate every
DP, ACT and DSRL model in one domain:

```bash
./run_eval.sh ideal                    # writes results/cotrain/<model>_ideal.json (existing files are skipped)
RESULTS=/tmp/check ./run_eval.sh real  # into another directory
```

**Train from scratch**

```bash
./run_cotrain.sh
```

This collects 300 sim and 100 real-stand-in demos into `data/` (several GB, about 12 MB per demo; scripted expert,
successful episodes only). It then trains the seven DP variants above (20k steps each, 4 running in parallel) and evaluates them in the
real stand-in. Each one trains on its own:

```bash
python dp.py train --alpha 0.5 --n_sim 300 --n_real 50 --steps 20000 --out runs/co_r50_a05.pt
```

**Record rollouts** (joint-state replay + an mp4 of the two cameras):

```bash
python record.py ../../models/dp/co_r50_a05.pt dp_co real 2000000,2000001,2000002
```

## 2. ACT

Action Chunking with Transformers. Same data, augmentation, mixing ratio and evaluation as DP. Each camera is a
ResNet-18 (GroupNorm) feature map giving 16 tokens. There is a CVAE encoder over the action chunk (32-d latent, KL
weight 10; z = 0 at run time), then 4 encoder and 7 decoder transformer layers with 16 queries. The model predicts
chunks of 16 actions every step and blends overlapping chunks by temporal ensembling (weights exp(-0.01 i)).

```bash
cd adamu_nut_yam/cotrain && source env.sh
python act.py eval --ckpt ../../models/act/act_co_r50_a05.pt --domain ideal --episodes 50 --json /tmp/act_ideal.json
./run_act.sh                       # train + evaluate the 4 ACT variants (needs data/ from run_cotrain.sh)
python act.py train --alpha 0.5 --n_real 50 --out runs/act_co_r50_a05.pt
python record.py ../../models/act/act_co_r50_a05.pt act_co real 2000000
```

The model type is recognised from the file name: names starting with `act_` load as ACT.

## 3. DP + DSRL

Diffusion Steering via RL, variant DSRL-NA, following the authors' `stable_baselines3/dsrl/dsrl.py`. The Diffusion
Policy is frozen. A small SAC policy instead chooses the initial noise (16 x 8) that the deterministic DDIM sampler
turns into an action chunk. It uses an action-space critic Q_A, a noise-space critic Q_W distilled from Q_A (noise
aliasing), and a tanh-Gaussian noise policy. The state is the frozen DP's observation encoding (1040-d). The reward is
sparse (1 on success). Hyperparameters are the repo's robomimic defaults: 3 x 2048 MLPs, UTD 20, action magnitude
1.5. Training runs online in the real stand-in for 1,500 episodes with 16 parallel environments.

```bash
cd adamu_nut_yam/cotrain && source env.sh
DSRL_DOMAIN=ideal python eval_dsrl.py --run ../../models/dsrl/dsrl_co_r50 --json /tmp/dsrl_ideal.json
python dsrl.py --base runs/co_r50_a05.pt --out runs/dsrl_co_r50 --episodes 1500      # train (~1.7 h)
./run_dsrl.sh                      # both runs (needs runs/co_r50_a05.pt and runs/real_r50.pt)
python record.py ../../models/dsrl/dsrl_co_r50 dsrl_co real 2000000
```

`models/dsrl/*/agent.pt` holds only the trained noise policy (the actor) plus its arguments. The frozen base it
steers is the DP named in those arguments (`models/dp/co_r50_a05.pt` or `real_r50.pt`); `--base` overrides it.
The critics are needed only to continue training and are not included. The learning curves are in
`results/cotrain/dsrl_*_training_log.jsonl`.

## 4. Touch in the Wild

Their pipeline, used unchanged: the UMI replay buffer and dataset classes, visuo-tactile MAE pretraining, and the UMI
diffusion-policy workspace (`DiffusionUnetTimmPolicy`, CLIP ViT-B/16 vision encoder) with their three tactile
options. Only the data source is new. `tw/sim.py` exposes the MuJoCo scene in UMI conventions:
* the YAM D405 wrist camera (224 px) replaces the GoPro;
* the FlexiTac pads give a 12 x 64 tactile image;
* gripper pose and width come from the simulator instead of SLAM;
* actions are absolute pinch-point poses executed with the arm IK.

`tw/collect.py` writes the same `.zarr.zip` replay buffer as their `07_generate_replay_buffer.py`. The scripted
demos start 30% of episodes with a missed grasp and 20% with the bump, and the expert recovers by re-grasping. They
are recorded through the release and retreat.

**Evaluate a released model** (uses `.venv-tw`)

```bash
source .venv-tw/bin/activate
export MUJOCO_GL=egl PYTHONPATH=$PWD/ext/touch_in_the_wild
python tw/eval_policy.py --ckpt models/tw/dp_tactile_mae.pt --cond bump --n 50 --out /tmp/tw_mae_bump.json \
    --record 3 --record_dir /tmp/tw_rollouts
```

`--cond` is `nominal` (the ideal condition) or `bump`. Rollouts include the wrist-camera frames, both tactile
images, joint states and the gripper command.

**Train from scratch** (about 7 h on one L40S: MAE 40 min, each policy about 1.9 h, evaluation 30 min). Set `OUT`
to a disk with ~20 GB free; every policy run keeps 2.6 GB of checkpoints, and the dataset is 1.5 GB:

```bash
OUT=/big/disk/tw tw/run_tw.sh
```

The script collects 300 demos (`tw/data/nut_tactile.zarr.zip`, 1.5 GB), runs MAE pretraining (30 epochs), trains
`dp_vision`, `dp_tactile_cnn` and `dp_tactile_mae` (25 epochs each, bf16), then evaluates each under nominal and
bump. To train only the MAE policy against the released encoder, run `MAE=$PWD/models/tw/visuotactile_mae.pth
tw/run_tw.sh`.

`models/tw/*.pt` are the EMA policy weights (fp32) with the training config. `eval_policy.py` also accepts the
full `checkpoints/latest.ckpt` from a training run. `models/tw/visuotactile_mae.pth` is the pretrained MAE (EMA
weights, in their checkpoint format). It was trained on an earlier 300-demo dataset from the same expert, without
the post-release frames or the bump demos; `run_tw.sh` pretrains it on the current dataset.

## 5. DexScrew

The three DexScrew stages on the tactile gripper, using their PPO / ProprioAdapt code (`ext/dexscrew`) on a
MuJoCo environment with the same observation contract as their IsaacGym task (`ds/nut_rl_env.py`):

* `obs` (34): recent gripper poses, last action, a noisy nut-position estimate (3 mm), nut yaw and the target.
* `priv_info` (11): exact nut pose relative to the gripper, tilt, nut mass and friction, pad friction, arm gain,
  pad contacts and latency.
* `proprio_hist` (30 x 24): proprioception plus 12 tactile features per step.
* Actions (5): Cartesian deltas, yaw rate and the grip command, executed with the arm IK. Training physics is
  randomised in every episode.

1. **Teacher** (privileged): actor-critic with their architecture (privileged MLP → 8-d latent → actor MLP
   512-256-128).
2. **Student**: their ProprioAdapt. A temporal-conv encoder learns to predict the teacher's latent from the 30-step
   proprio + touch history, reusing the teacher's actor (10 M steps, 224 environments).
3. **Tactile BC fine-tune**: the student, with exploration noise, collects 400 episodes in the shifted-physics
   stand-in for the real robot. The 130 successful ones become behaviour-cloning data for the actor and the temporal
   encoder. The paper instead uses human teleoperation on the real robot, assisted by the stage-2 skill.

**Deviation from the paper: the teacher is not trained with PPO.** PPO from scratch (`train_ds.py teacher`, their
hyperparameters) reached 0% success after three reward designs. It learned to hover holding the nut, to stall
6 cm short because of integrator wind-up, and to farm the reach/alignment shaping without grasping. The teacher is
instead trained to imitate the scripted controller (`heuristic.py`, 100% on randomised physics) with DART noise on
the executed actions (`bc_teacher.py`). Extra DAgger rounds (`--rounds > 1`) made it worse (34/50 nominal after 5
rounds), because the expert's labels depend on its hidden phase counters. `results/ds/teacher_dagger_log.json` has
that run. Stages 2 and 3 follow DexScrew.

**Evaluate a released model** (uses `.venv-tw`)

```bash
source .venv-tw/bin/activate && export MUJOCO_GL=egl && cd ds
python eval_ds.py --ckpt ../models/ds/stage3_ft.ckpt --kind student --cond shifted --n 50 --out /tmp/ft_shifted.json \
    --record 3 --record_dir /tmp/ds_rollouts --tag ft
python eval_ds.py --ckpt ../models/ds/teacher.pth --kind teacher --cond nominal --n 50 --out /tmp/teacher.json
```

`--kind` is `teacher` for `teacher.pth` and `student` for `student.ckpt` and `stage3_ft.ckpt`. `--cond` is `nominal`
(ideal), `bump` or `shifted`.

**Train from scratch** (about 2 h: teacher 5 min, student 1.3 h at ~2,000 steps/s on 28 CPU processes, stage 3
5 min, evaluation 20 min):

```bash
ds/run_ds.sh                     # OUT=<dir> to put training outputs elsewhere (default ds/runs)
```

## Interactive evaluation page (`webapp/`)

A local web page for trying the released models on layouts you choose. You drag the nut (and set its yaw) and the
target pad on a top-down map of the table; the dashed boxes show the training distribution. Then you pick any of the
19 models and a condition, set a seed and press **Pick**. The episode runs in MuJoCo on the GPU and plays back live
in a 3D view with readouts, the camera image the policy saw and the FlexiTac pad heatmaps. Runs are kept in a list
for replay.

```bash
webapp/run.sh            # http://localhost:8770 on the server (listens on localhost only)
ssh -L 8770:localhost:8770 <user>@<server>     # then open http://localhost:8770 on your laptop
```

How it works:
* `server.py` (stdlib HTTP, `.venv`) serves the page and a small JSON API, and queues episodes.
* Two `worker.py` processes keep the models on the GPU: `cam` in `.venv` for DP, ACT and DSRL, and `tac` in
  `.venv-tw` for Touch in the Wild and DexScrew. Each streams one line per control step: body poses, gripper,
  camera JPEG and tactile image.
* The layout reaches the simulator through the `layout` override in `NutPickPlaceEnv.reset`
  (`adamu_nut_yam/env.py`), which every environment inherits. Everything else in the episode comes from the seed,
  and the success test is the one used offline.
* `build_assets.py` exports the two scenes as browser geometry (run automatically on first start).

An episode takes 2 to 15 s of compute: DexScrew ~2 s, DP / ACT / DSRL 3 to 10 s, Touch in the Wild ~10 to 15 s.
The first run of each model adds a few seconds to load it. The workers fall back to the CPU when there is no GPU
(set `MUJOCO_GL=osmesa` for software rendering).

### Public hosting: Hugging Face Space + GitHub Pages

| Piece | Where | Made by |
|---|---|---|
| Models and results | [`josyula/adamu-nut-policies`](https://huggingface.co/josyula/adamu-nut-policies) | `hf upload` of `models/`, `results/`, `rollouts/` |
| Model server (this page's API) | [`spaces/josyula/adamu-nut-eval`](https://huggingface.co/spaces/josyula/adamu-nut-eval), Docker | `space/make_space.sh --upload` |
| Static page | GitHub Pages | `pages/make_pages.sh <dir>`, then push `<dir>` |

The Space image (`space/Dockerfile`) installs both environments with CPU (or, with `--build-arg TORCH=cu121`, CUDA)
wheels, clones the two upstream repositories, downloads the models from the model repo and runs `webapp/run.sh` on
port 7860 with software rendering. The server sends `Access-Control-Allow-Origin` (`ALLOW_ORIGIN`, default `*`) so
the GitHub Pages copy can call it, and refuses new episodes with HTTP 429 once `MAX_QUEUE` (8) are waiting for a
worker. The Pages copy is the same `index.html` with `<meta name="eval-api">` set to the Space URL; `?api=<url>` on
the page URL points it at any other server.

On the free CPU Space (2 vCPU) an episode takes about 5 to 10 s for DP, ACT, DSRL and DexScrew and about a minute
for Touch in the Wild, and the Space sleeps after 48 hours without visitors (the page waits while it wakes). For GPU
speed, switch the Space hardware to a T4 and rebuild with `TORCH=cu121`.

---

## Trained models (`models/`)

| File | Method | Size |
|---|---|---|
| `dp/{sim_only,real_r10,real_r50,real_r100,co_r10_a025,co_r10_a05,co_r50_a05}.pt` | Diffusion Policy | 93 MB each |
| `act/{act_sim_only,act_real_r50,act_co_r10_a05,act_co_r50_a05}.pt` | ACT | 89 MB each |
| `dsrl/{dsrl_co_r50,dsrl_real_r50}/agent.pt` | DSRL noise policy (base DP in `dp/`) | 22 MB each |
| `tw/{dp_vision,dp_tactile_cnn,dp_tactile_mae}.pt`, `tw/visuotactile_mae.pth` | Touch in the Wild | 0.7 to 1.2 GB; MAE 185 MB |
| `ds/{teacher.pth,student.ckpt,stage3_ft.ckpt}` | DexScrew stages 1 to 3 | 1 MB each |

DP, ACT, DSRL and the MAE encoder are stored in fp16 (small tensors such as normalisation statistics stay fp32) and
are cast back to fp32 on load. The Touch in the Wild policies are stored in fp32: the EMA weights and the training
config, without optimiser state.

**Released-model check.** Every released model was re-evaluated from this repository under the ideal condition
(50 seeds):

| Model | Successes, original | Successes, released file |
|---|---|---|
| DP: sim_only, co_r10_a025, co_r10_a05, co_r50_a05 | 40, 22, 17, 32 | 38, 23, 17, 30 |
| DP real only (r10/r50/r100) | 0 | 0 |
| ACT: sim_only, real_r50, co_r10_a05, co_r50_a05 | 7, 7, 7, 9 | 7, 6, 7, 9 |
| DSRL: dsrl_co_r50, dsrl_real_r50 | 36, 0 | 34, 0 |
| Touch in the Wild: vision, tactile_cnn, tactile_mae | 38, 33, 33 | 35, 32, 34 |
| DexScrew: teacher, student, stage-3 fine-tune | 50, 40, 25 | 50, 40, 25 |

The DP, ACT and DSRL evaluations fix the diffusion noise per seed, so their small differences come from fp16. The
Touch in the Wild evaluator does not seed the diffusion sampling, so a repeat run of the same checkpoint varies by a
few episodes (for example, the original vision checkpoint scored 38 and then 39).

## Rollouts (`rollouts/`)

| Folder | Content |
|---|---|
| `cotrain/` | DP (`dp_sim`, `dp_co`, `dp_real` = real_r100), ACT (`act_co`) and DSRL (`dsrl_co`), seeds 2000000 to 2000002, real stand-in and ideal: `<model>_<domain>_<seed>.npz` (joint positions per step, gripper command, nut position, success) and `_cams.mp4` (head and wrist camera, what the policy saw) |
| `tw/` | Touch in the Wild: `<model>_<cond>_<seed>.npz` with `rgb` (wrist camera), `tact` (12 x 64 tactile image), `qpos`, `grip`, `target`, `meta` |
| `ds/` | DexScrew: `<stage>_<cond>_<seed>.npz` with `qpos`, `tact` (2 x 32 x 12), `grip`, `success` |

`qpos` is the full MuJoCo state vector of the matching scene (`adamu_nut_yam/assets/scene.xml` or
`adamu_nut_tactile/assets/scene.xml`). Set it on `MjData` and call `mj_forward` to replay or re-render an episode.
For DexScrew, three seeds per condition are recorded, plus two successful seeds under shifted physics.

## Notes

* **Real-only policies score 0% in the ideal domain.** They never saw the default appearance, and they learned to
  act around the stand-in's 150 ms latency. Co-training keeps most of the sim performance and gains the stand-in.
* **Touch gives no gain in these conditions.** The wrist camera sees the bump clearly; only the MAE encoder matches
  camera-only when the nut is bumped.
* **DexScrew's student has no camera.** Once the bump pushes the nut out of the jaws, the touch history alone cannot
  relocate it (0%).
* **Seeds:** demonstrations use seeds from 0 (sim) and 1,000,000 (real stand-in); DexScrew training uses 3,000,000+
  and 5,000,000+; evaluation always uses 2,000,000+.

Third-party code: Touch in the Wild (MIT, Columbia RoboPIL) and DexScrew (MIT) are cloned, not vendored. The ADAM-U
model comes from PNDbotics' `pnd_mujoco` and the YAM gripper meshes from MuJoCo Menagerie (`i2rt_yam`). The FlexiTac
sensor model follows `binghao-huang/FlexiTac-IsaacSim-Simulation`.
