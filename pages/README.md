# ADAM-U nut pick: interactive evaluation page

Static front end for evaluating trained robot policies in simulation. Drag the nut and the target on the table
map, choose a policy (Diffusion Policy, ACT, DP + DSRL, Touch in the Wild, DexScrew) and a condition, and press
**Pick**. The episode runs in MuJoCo on a Hugging Face Space and streams back to this page: the robot in 3D, the
camera image the policy saw and, for the tactile policies, the FlexiTac pads.

* Models: [huggingface.co/josyula/adamu-nut-policies](https://huggingface.co/josyula/adamu-nut-policies)
* Model server: [huggingface.co/spaces/josyula/adamu-nut-eval](https://huggingface.co/spaces/josyula/adamu-nut-eval)

The page is plain HTML with Three.js from a CDN; there is no build step. The server URL is the
`<meta name="eval-api">` tag in `index.html`, and `?api=<url>` on the page URL overrides it (for example to point
at a server running on your own machine: `?api=http://localhost:8770`).

`scene_*.json` and `meshes_*.bin` are the robot and table geometry, exported from the MuJoCo scenes.

## How hosted results can differ from the published ones

The success rates shown on the page (and in the model card) were measured on an NVIDIA GPU with the full-detail
renderer, on 50 held-out seeds with the training layout distribution. Episodes run from this page can score
differently, because:

* **CPU instead of GPU.** The hosted server has no GPU, so the policies run in fp32 on the CPU instead of bf16 on the
  GPU, and small numerical differences can change the diffusion sampling and therefore the trajectory.
* **Software rendering.** Camera images are rendered with Mesa llvmpipe (OSMesa) instead of NVIDIA EGL, which
  shades and anti-aliases slightly differently.
* **Simplified meshes for the cameras.** To make CPU rendering fast enough, the robot meshes used for the camera
  images are decimated to at most 2,000 triangles each (from 1.4 M in total) and the shadow maps are 1024 px instead
  of 2048 px. Physics is unchanged. The images differ from the originals by about 1.5 / 255 per pixel on average,
  slightly more on edges.
* **Compressed weights.** Diffusion Policy, ACT and DSRL checkpoints are stored in fp16 (re-evaluated: within 2
  successes out of 50 of the originals).
* **Unseeded sampling.** Touch in the Wild does not fix its diffusion noise, so the same seed and layout can give a
  different result on another run.
* **Your layout.** Nut or target positions outside the dashed training regions (or the nut very close to the
  target) are outside what the policies were trained on.

These effects were not measured end to end on the hosted server, so treat single episodes as illustrations, not as a
new success-rate estimate.
