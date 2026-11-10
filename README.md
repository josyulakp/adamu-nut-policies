# ADAM-U nut pick-and-place: rollout and results viewer

This branch is the GitHub Pages site for
[josyulakp/adamu-nut-policies](https://github.com/josyulakp/adamu-nut-policies):
**https://josyulakp.github.io/adamu-nut-policies/**

It replays the offline rollouts in 3D (MuJoCo body poses for every 50 ms control step, with the FlexiTac pad
readings and wrist-camera video where the policy used them) and shows the evaluation results table:

| Group | Episodes |
|---|---|
| Dexterous hand: scripted expert, BC on held-out seeds | `expert_*`, `bc_*` |
| YAM gripper: scripted expert, BC | `yam_expert_*`, `yam_bc_*` |
| YAM, real-robot stand-in: Diffusion Policy (sim only, co-trained), ACT, DSRL | `yam_dp_sim_*`, `yam_dp_co_*`, `yam_act_co_*`, `yam_dsrl_co_*` |
| FlexiTac YAM: Touch in the Wild (camera only, SimpleCNN, MAE), nominal and bump | `tw_*` |
| FlexiTac YAM: DexScrew teacher, student, tactile fine-tune; nominal, bump, shifted physics | `ds_*` |

Files: `index.html` (the page, no build step), `episodes.json` (catalogue), `episodes/<id>.json` (joint states,
readouts, tactile images) and `<id>.b64.txt` (precomputed body poses), `results.json` (results table),
`scene*.json` / `meshes*` (geometry for the three end-effector variants) and `mujoco/` (MuJoCo 3.17 WebAssembly and
scene files for the optional in-browser engine).
