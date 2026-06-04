# LeRealWorldModel

<video src="https://github.com/Maelic/LeRealWorldModel/raw/main/figs/decoder_recon_sidebyside.mp4" controls width="640"></video>

A **JEPA latent world model** + **GC-IDM amortized planner** for goal-conditioned
manipulation on the [SO-100](https://github.com/TheRobotStudio/SO-ARM100) arm —
built directly on [LeRobot](https://github.com/huggingface/lerobot) and
[stable-worldmodel](https://github.com/rbalestr-lab/lewm).

The clip above is a probe into what the model actually learns: every frame's
192-d JEPA `CLS` token decoded back to pixels (left: ground truth, right:
decoded from the latent).

## What it is

Train a world model on a real-robot LeRobot dataset, then drive the arm toward a
**goal image** by planning in latent space. Three layers, cleanly separated:

```
LeRobot           hardware interface, data collection, deployment   (unmodified)
lewm_robot        this repo: data adapter, training, planner, deploy
stable-worldmodel JEPA primitives, losses, solver utilities          (unmodified)
```

The world model is **not a behavioural policy** — it predicts future latents
conditioned on actions. A planner turns it into control: frames in, actions out.

Two planners are implemented:

- **GC-IDM** (current, `lewm_robot/`) — a Goal-Conditioned Inverse Dynamics MLP
  that maps `(zₜ, z_goal, horizon) → action` in a single forward pass, replacing
  CEM/MPPI search for ~100× faster closed-loop control.
- **Random-shooting / CEM MPC** (legacy, `lewm/`) — samples action chunks,
  rolls them out in latent space, and picks the chunk whose predicted latent is
  closest to the goal.

## How it works

- **World model (JEPA).** ViT-Tiny encoder → 192-d `CLS`, an autoregressive
  predictor, and an action embedder. Trained with latent-space predictor MSE +
  SIGReg regulariser — **no pixel reconstruction loss**.
- **Action representation.** `frameskip = 5` chunked actions, so the effective
  action dim is `frameskip × robot_dof = 30` for SO-100. The predictor works in
  the chunked space; the real robot is driven at native fps.
- **Planner (GC-IDM).** A small MLP with AdaLN-Zero horizon conditioning,
  trained by supervised regression on frozen encoder embeddings (Stage 2).

## Install

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

Python ≥3.12 is required by upstream LeRobot. LeRobot and stable-worldmodel are
editable installs — no source modifications are needed; the JEPA policy is
discovered through LeRobot's plugin path.

## Pipeline

```
collect_data ─▶ Stage 1: JEPA world model ─▶ Stage 2: GC-IDM planner ─▶ export ─▶ deploy
                        │
                        └─▶ analysis: identifiability suite + pixel decoder
```

### 1. Collect demonstrations (optional)

Teleoperate the SO-100 (leader → follower) with two cameras (`up`, `side`):

```bash
./scripts/collect_data.sh 20 maelicneau/stack_cubes "Stack three cubes."
```

Saves a LeRobot dataset to `./datasets/stack_cubes`. You can also use any
existing LeRobot dataset.

### 2. Stage 1 — JEPA world model

```bash
./scripts/train_stage1.sh lewm_so100_topcam      # top camera only, 50 epochs
# ./scripts/train_stage1.sh lewm_so100_dualcam   # top + side fused, 50 epochs
# ./scripts/train_stage1.sh lewm_so100           # dual cam, 100 epochs (default)
```

Produces, in the run directory, `lewm_*_epoch_N_object.ckpt` (pickled JEPA, used
by Stage 2 and deploy), `*_normalizers.pt` (per-joint action mean/std), and a
`*.safetensors` export.

### 3. Stage 2 — GC-IDM planner *(optional)*

> **Skip this step to use CEM planning instead** — see the fallback below.

GC-IDM ([*Latent Geometry Beyond Search*](https://arxiv.org/abs/2605.08732))
replaces CEM's expensive sample-and-score loop with a single MLP forward pass:
`(zₜ, z_goal, horizon) → action`. The result is ~100× faster inference, making
closed-loop control on hardware practical without a dedicated GPU budget for
planning.

Point it at the Stage 1 checkpoint; the config is auto-selected:

```bash
./scripts/train_stage2.sh checkpoints/so100_topcam/lewm_so100_topcam_epoch_50_object.ckpt
```

Pre-computes all frozen-encoder embeddings, then trains by MSE (~20 min, single
GPU). Writes `gc_idm.pt` next to the checkpoint.

**Fallback — CEM planning (no Stage 2 required).** If you skip Stage 2, the
legacy `lewm/` planner samples action chunks, rolls them out in latent space, and
picks the chunk closest to the goal in embedding space:

```bash
python -m lewm.deploy_so100 \
    --ckpt        checkpoints/so100_topcam/lewm_so100_topcam_epoch_50_object.ckpt \
    --normalizers checkpoints/so100_topcam/lewm_so100_topcam_normalizers.pt \
    --goal-image  ./goal.png \
    --port /dev/ttyACM0 --camera-key observation.images.up \
    --horizon 8 --num-samples 256 --fps 30 --max-steps 300
```

CEM is slower per step (~256 latent rollouts each tick) but requires no extra
training and can be useful for debugging or as a reference baseline.

### 4. Sanity-check the model

**Identifiability suite** — affine/nonlinear probes, action diversity, temporal
contrastivity, equivariance, action invertibility, probe generalisation, DCI,
plus an action-corruption ablation (`--corruption`):

```bash
python run_identifiability_so100.py \
    --ckpt        checkpoints/so100_topcam/lewm_so100_topcam_epoch_50_object.ckpt \
    --normalizers checkpoints/so100_topcam/lewm_so100_topcam_normalizers.pt
```

**Pixel-decoder probe** — train a lightweight decoder on the frozen `CLS` token
so you can *see* what the latent encodes (reconstruction, ground-truth-action
rollout, and GC-IDM-planned rollout):

```bash
./scripts/train_decoder.sh checkpoints/so100_topcam
# → decoder_recon.png, decoder_rollout.png, decoder_gcidm_rollout.png

# Side-by-side GT | decoded video (the teaser above):
python scripts/make_recon_gif.py \
    --world-model-path checkpoints/so100_topcam/lewm_so100_topcam_epoch_50_object.ckpt \
    --decoder-path     checkpoints/so100_topcam/decoder.pt \
    --out figs/decoder_recon_sidebyside.gif --episode 0 --num-frames 120 --fps 15 --mp4
```

### 5. Export a LeRobot checkpoint

Bundle the world model + GC-IDM into a standard LeRobot policy directory:

```bash
./scripts/export_policy.sh checkpoints/so100_topcam [goal.jpg]
# → checkpoints/jepa_so100/  (config.json + gc_idm.pt + processors)
```

### 6. Deploy on SO-100

Dry-run first (replays a dataset, runs the planner each tick, sends nothing to
hardware):

```bash
python -m lewm_robot.deploy_jepa_so100 \
    --world-model-path checkpoints/so100_topcam/lewm_so100_topcam_epoch_50_object.ckpt \
    --gc-idm-path      checkpoints/so100_topcam/gc_idm.pt \
    --goal-image ./goal.jpg --image-keys observation.images.up \
    --dry-run-replay-from maelicneau/stack_cubes \
    --dry-run-replay-root ./datasets/stack_cubes
```

Then on hardware — capture a goal from the live cameras and run closed-loop:

```bash
FPS=6 MAX_RELATIVE_TARGET=8 \
./scripts/deploy_jepa.sh checkpoints/so100_topcam --capture-goal --horizon-floor 10
```

> **Keep a hand on the e-stop.** GC-IDM predicts *absolute* joint targets at the
> frameskip-decimated rate (≈ `dataset_fps / frameskip` ≈ 6 Hz), so run the loop
> near that rate rather than the camera fps, and cap per-step motion with
> `MAX_RELATIVE_TARGET`. `--horizon-floor` stops the planner collapsing to a
> single-step "lunge to goal" once the horizon runs out.

An alternative path through LeRobot's own rollout harness is available via
`./scripts/deploy_lerobot_rollout.sh` (registers `JEPAPolicy` as a plugin so it
benefits from Sentry / RTC strategies).

## Repo layout

```
.
├── jepa.py / module.py / utils.py     # JEPA model, predictor/embedder/SIGReg, helpers
├── train_lewm.py                      # Stage 1 entry-point (Hydra + Lightning)
├── train_gc_idm.py                    # Stage 2: GC-IDM supervised training
├── export_policy.py                   # bundle → LeRobot checkpoint
├── train_jepa_decoder.py              # pixel-decoder probe (CLS → image)
├── train_decoder.py                   # older standalone decoder probe
├── identifiability.py                 # probes, equivariance, DCI, action diversity
├── action_diversity.py               # corrupt_actions ablation
├── run_identifiability_so100.py       # identifiability eval runner (SO-100)
├── config/train/                      # lewm_so100*.yaml (Stage 1), gc_idm*.yaml (Stage 2)
├── scripts/                           # collect / train_stage{1,2} / export / deploy / decoder
├── lewm_robot/                        # current package: JEPA + GC-IDM
│   ├── policies/jepa/                 # JEPAConfig, JEPAPolicy, GCIDM, processor
│   ├── decoder.py                     # JEPADecoder (MAE-style)
│   ├── deploy_jepa_so100.py           # standalone closed-loop deploy
│   └── rollout_jepa.py                # LeRobot-rollout integration
└── lewm/                              # legacy CEM / random-shooting planner + deploy
```

## Caveats before trusting hardware

`run_identifiability_so100.py` is the source of truth for whether a model is
ready to deploy. Watch for:

- **Action dependence < 0.1** — the predictor is ignoring the action; any
  planner built on it will degenerate. Train longer or with more action-diverse
  data. (Use `--corruption` to confirm the metric responds to action scrambling.)
- **Action effective rank ≪ effective action dim** — the *dataset* lacks action
  diversity. Collect more demos or co-train on additional SO-100 datasets.
- **Probe test R² ≪ probe train R²** — the encoder overfits a small set of
  episodes and won't generalise to a live camera frame.

Before motors touch anything, confirm action coordinate-frame parity by
replaying a recorded demo through `robot.send_action()` *without the model* and
checking the arm reproduces the trajectory.

## Acknowledgments

- **[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)** (Maes et al., 2026) —
  JEPA model primitives, SIGReg loss, and the original training loop this repo
  builds on.
- **[LeRobot](https://github.com/huggingface/lerobot)** (Hugging Face) —
  hardware interface, dataset format, and deployment harness; used entirely
  unmodified.
- **[Latent Geometry Beyond Search: Amortizing Planning in World Models](https://arxiv.org/abs/2605.08732)**
  (Nguyen et al., 2026) — the GC-IDM architecture and training recipe that
  replaces CEM with a single amortized MLP forward pass. Code reference:
  [hdnndh/Latent-Geometry-Beyond-Search-Amortizing-Planning-in-World-Models](https://github.com/hdnndh/Latent-Geometry-Beyond-Search-Amortizing-Planning-in-World-Models).

## License

Apache-2.0, inherited from upstream le-wm. See `LICENSE`.
