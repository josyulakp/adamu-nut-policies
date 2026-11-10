#!/usr/bin/env python3
"""DexScrew pipeline (stages 1-2) on the MuJoCo FlexiTac nut task, using the repo's own PPO and ProprioAdapt.

  teacher:  python train_ds.py teacher --steps 40e6 --out ds/runs/teacher_ppo   (did not converge on this task; see README)
  student:  python train_ds.py student --teacher <.../stage1_nn/best*.pth> --steps 10e6 --out .../student

Hyperparameters follow configs/train/XHandHoraNutBolt.yaml (MLP [512,256,128], priv MLP [256,128,8] -> 8-d latent,
gamma 0.99, GAE 0.95, lr 5e-3 with KL-adaptive schedule (0.02), 5 mini-epochs, critic coef 4, clip 0.2, no entropy
bonus), scaled to 224 CPU environments: horizon 32 (their 12 with 16k GPU envs), minibatch 1792.
Point clouds are off (the nut's shape is fixed; its pose is in the privileged vector).
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "ext" / "dexscrew"))

from omegaconf import OmegaConf  # noqa: E402


def make_cfg(n_envs, steps, proprio_adapt, horizon=32):
    return OmegaConf.create({
        "rl_device": "cuda:0", "test": False,
        "train": {
            "network": {"mlp": {"units": [512, 256, 128]}, "priv_mlp": {"units": [256, 128, 8]},
                        "point_mlp": {"units": [32, 32, 32]}},
            "ppo": {"normalize_input": True, "normalize_value": True, "normalize_priv": True,
                    "normalize_point_cloud": False, "value_bootstrap": True, "num_actors": n_envs,
                    "gamma": 0.99, "tau": 0.95, "learning_rate": 5e-3, "kl_threshold": 0.02,
                    "horizon_length": horizon, "minibatch_size": n_envs * horizon // 4, "mini_epochs": 5,
                    "clip_value": True, "critic_coef": 4, "entropy_coef": 0.0, "e_clip": 0.2,
                    "bounds_loss_coef": 0.0, "truncate_grads": True, "grad_norm": 1.0, "save_best_after": 0,
                    "save_frequency": 50, "max_agent_steps": int(steps), "priv_info": True,
                    "use_point_cloud_info": False, "proprio_len": 30, "proprio_dim": 24,
                    "proprio_adapt": proprio_adapt},
        }})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["teacher", "student"])
    ap.add_argument("--steps", type=float, default=40e6); ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", default=""); ap.add_argument("--init", default="")
    ap.add_argument("--init_logstd", type=float, default=-10.0); ap.add_argument("--procs", type=int, default=28)
    ap.add_argument("--per_proc", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    torch.manual_seed(a.seed)
    from vec_env import MjVecEnv
    env = MjVecEnv(n_procs=a.procs, per_proc=a.per_proc, seed=a.seed * 100_000)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.stage == "teacher":
        from dexscrew.algo.ppo.ppo import PPO
        agent = PPO(env, a.out, make_cfg(env.num_envs, a.steps, False))
        if a.init:                       # warm start (e.g. after a reward fix); re-open the action std
            agent.restore_train(a.init)
            import torch
            ck = torch.load(a.init, weights_only=False)
            if "value_mean_std" in ck:
                agent.value_mean_std.load_state_dict(ck["value_mean_std"])
            with torch.no_grad():
                agent.model.sigma.clamp_(min=a.init_logstd)
        agent.train()
    else:
        from dexscrew.algo.ppo.padapt import ProprioAdapt
        agent = ProprioAdapt(env, a.out, make_cfg(env.num_envs, a.steps, True))
        agent.restore_train(a.teacher)
        env.step_budget = int(a.steps)
        try:
            agent.train()
        except StopIteration:
            pass
        agent.save(str(Path(a.out) / "stage2_nn" / "model_last"))
    print("stats", env.stats, flush=True)
    env.close()


if __name__ == "__main__":
    main()
