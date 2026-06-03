# LeRealWorldModel

Train a **JEPA-style latent world model** ([le-wm](https://github.com/rbalestr-lab/lewm)) directly on a
[LeRobot](https://github.com/huggingface/lerobot) dataset, then deploy it on a
real robot (SO-100) with random-shooting / CEM MPC.

The repo wires three things together:

1. **`jepa.py` + `train.py`** — the baseline lewm JEPA model (ViT encoder + AR
   predictor + action embedder), trained via Hydra + Lightning. Loss = predictor
   MSE in latent space + SIGReg regularizer. No reconstruction loss.
2. **`lewm/data/lerobot_adapter.py`** — adapter that exposes a
   `LeRobotDataset` (parquet + mp4 + meta.json) through the
   `stable_worldmodel.data.Dataset` API, so the existing training loop runs
   on real-robot data without conversion to HDF5.
3. **`lewm/deploy_so100.py` + `lewm/policies/wm_planning/`** — two deploy
   paths:
   - **standalone**: `python -m lewm.deploy_so100 ...` imports `SOFollower`
     directly and runs random-shooting MPC against a goal image;
   - **LeRobot plugin**: `python -m lewm.rollout_wm_planning ...` registers
     `WMPlanningPolicy` as `--policy.type=wm_planning` and forwards to
     `lerobot-rollout` so it benefits from Sentry / RTC strategies.

## What the world model does (and doesn't)

The world model is **not a behavioural policy.** It predicts future latents
conditioned on actions. Deployment wraps it with a planner that samples
candidate action chunks, simulates them in latent space via
`JEPA.rollout`, scores them by cosine similarity to a goal latent, and sends
the best first-chunk to the robot. Frames in/actions out, no policy network.

Action representation: **frameskip = 5 chunked actions** (effective action
dim = `frameskip × robot_dof` = 30 for SO-100). The predictor and embedder
work on the chunked space; the real robot is driven at native fps with the
unchunked actions.

## Installation

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

Python ≥3.12 is required by the upstream LeRobot package.

## Quick start

### 1. Train on `lerobot/svla_so100_pickplace`

```bash
python train.py data=lerobot \
    data.dataset.repo_id=lerobot/svla_so100_pickplace \
    data.dataset.image_key=observation.images.top \
    wm.action_dim=6 \
    loader.batch_size=32 \
    compile=false wandb.enabled=false
```

Outputs land in `~/.stable_worldmodel/<run_id>/`:

- `lewm_epoch_N_object.ckpt` — pickled JEPA module per epoch (used by the deploy script)
- `lewm_weights.ckpt` — Lightning state dict
- `lewm_normalizers.pt` — per-column `(mean, std)` for action denormalisation

### 2. Sanity-check the trained model

Identifiability suite: equivariance, k-step rollout, linear/MLP probe on
proprio, action diversity, action invertibility, DCI:

```bash
python run_identifiability_so100.py \
    --ckpt ~/.stable_worldmodel/lewm_epoch_19_object.ckpt \
    --normalizers ~/.stable_worldmodel/lewm_normalizers.pt
```

Pixel-decoder probe (trains a small cross-attention decoder on top of the
frozen CLS embedding so you can *see* what the latent encodes):

```bash
python train_decoder.py \
    --checkpoint ~/.stable_worldmodel/lewm_epoch_19_object.ckpt \
    --lerobot-repo lerobot/svla_so100_pickplace \
    --lerobot-normalizers ~/.stable_worldmodel/lewm_normalizers.pt \
    --epochs 50 --batch-size 64 \
    --output-dir ./decoder_weights/so100

# Visualise imagined rollouts
python train_decoder.py \
    --checkpoint ~/.stable_worldmodel/lewm_epoch_19_object.ckpt \
    --lerobot-repo lerobot/svla_so100_pickplace \
    --lerobot-normalizers ~/.stable_worldmodel/lewm_normalizers.pt \
    --decoder-weights ./decoder_weights/so100/decoder_best.pt \
    --visualize --num-rollouts 8 --rollout-steps 15
```

### 3. Dry-run the deploy (no hardware)

Replays observations from the dataset, runs the planner each tick, logs the
chosen actions but does *not* send them to a robot.

```bash
python -m lewm.deploy_so100 \
    --ckpt ~/.stable_worldmodel/lewm_epoch_19_object.ckpt \
    --normalizers ~/.stable_worldmodel/lewm_normalizers.pt \
    --goal-image ./goal.png \
    --camera-key observation.images.top \
    --dry-run-replay-from lerobot/svla_so100_pickplace \
    --max-steps 60
```

### 4. Real-world deploy on SO-100

```bash
python -m lewm.deploy_so100 \
    --ckpt ~/.stable_worldmodel/lewm_epoch_19_object.ckpt \
    --normalizers ~/.stable_worldmodel/lewm_normalizers.pt \
    --goal-image ./goal.png \
    --port /dev/ttyACM0 \
    --camera-key observation.images.top \
    --camera-index 0 --camera-width 640 --camera-height 480 \
    --history-size 3 --horizon 8 --num-samples 256 \
    --frameskip 5 --fps 30 \
    --max-relative-target 5 \
    --max-steps 30
```

Keep your hand on the e-stop. `--max-relative-target 5` clamps each
joint command to ≤5° from the current pose so a wildly wrong action can't
lurch the arm.

Alternative path through the LeRobot rollout harness (uses `WMPlanningPolicy`
plugin):

```bash
python -m lewm.rollout_wm_planning \
    --strategy.type=base \
    --policy.type=wm_planning \
    --policy.world_model_path=~/.stable_worldmodel/lewm_epoch_19_object.ckpt \
    --policy.normalizers_path=~/.stable_worldmodel/lewm_normalizers.pt \
    --policy.goal_image_path=./goal.png \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras='{"top":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30}}' \
    --task="pick up the cube" --duration=30
```

## Repo layout

```
.
├── jepa.py                       # JEPA model (encode, predict, rollout, get_cost)
├── module.py                     # ARPredictor, Embedder, MLP, SIGReg
├── utils.py                      # img preprocessor, column normaliser, callbacks
├── train.py                      # Hydra + Lightning training entry-point
├── train_decoder.py              # Pixel-decoder probe (HDF5 and LeRobot)
├── identifiability.py            # MCC, affine/nonlinear probes, DCI, etc.
├── action_diversity.py           # action-corruption utilities
├── run_identifiability_so100.py  # eval suite for SO-100 / LeRobot models
├── run_identifiability_pusht.py  # eval suite for HDF5 pusht models
├── config/
│   └── train/
│       ├── lewm.yaml             # default training config
│       └── data/
│           ├── lerobot.yaml      # LeRobotWMDataset config
│           └── pusht.yaml        # HDF5 reference (pusht)
└── lewm/
    ├── data/lerobot_adapter.py             # LeRobotDataset → stable_worldmodel.Dataset
    ├── planning/mpc.py                     # RandomShootingPlanner, CEMPlanner
    ├── deploy_so100.py                     # standalone real-world deploy
    ├── rollout_wm_planning.py              # wrapper around lerobot-rollout
    └── policies/wm_planning/               # PreTrainedPolicy plugin
        ├── configuration_wm_planning.py
        ├── modeling_wm_planning.py
        └── processor_wm_planning.py
```

## Caveats and things to verify before trusting hardware

The identifiability eval (`run_identifiability_so100.py`) is the source of
truth for whether a model is ready to deploy. Watch for:

- **Action dependence < 0.1**: the predictor is ignoring the action. MPC
  will degenerate. Train longer or with more action-diverse data.
- **Action effective rank ≪ effective_act_dim**: the *dataset* lacks action
  diversity. Collect more demos or co-train on additional SO-100 datasets.
- **Probe test R² ≪ probe train R²**: the encoder overfits to the small
  set of episodes and won't generalise to a live camera frame.

Before motors touch anything, also confirm action coordinate-frame parity
by replaying a recorded demo through `robot.send_action()` *without the
model* and confirming the arm reproduces the trajectory.

## License

Apache-2.0, inherited from upstream le-wm. See `LICENSE`.
