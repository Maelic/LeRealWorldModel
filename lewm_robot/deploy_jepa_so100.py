"""Real-time GC-IDM deployment on a dual-camera SO-100 arm.

This script runs JEPAPolicy (JEPA world model + GC-IDM amortized planner)
directly against the SO-100 hardware at up to 30 Hz closed-loop.

Usage — capture goal from live cameras, then run::

    python -m lewm_robot.deploy_jepa_so100 \\
        --world-model-path ~/.stable_worldmodel/<run>/lewm_so100_epoch_100_object.ckpt \\
        --gc-idm-path ~/.stable_worldmodel/<run>/gc_idm.pt \\
        --normalizers-path ~/.stable_worldmodel/<run>/lewm_so100_normalizers.pt \\
        --port /dev/ttyACM0 \\
        --camera-up-index 0 \\
        --camera-side-index 2 \\
        --capture-goal

Usage — load goal from file::

    python -m lewm_robot.deploy_jepa_so100 \\
        --world-model-path ... \\
        --gc-idm-path ... \\
        --goal-image ./goal.jpg

Dry-run (replay observations from a LeRobot dataset, no hardware)::

    python -m lewm_robot.deploy_jepa_so100 \\
        --world-model-path ... --gc-idm-path ... --goal-image ./goal.jpg \\
        --dry-run-replay-from maelicneau/stack_cubes \\
        --dry-run-replay-root /path/to/datasets/stack_cubes
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import lewm_robot  # noqa: F401 — registers JEPAConfig

from lewm_robot.policies.jepa.configuration_jepa import JEPAConfig
from lewm_robot.policies.jepa.modeling_jepa import JEPAPolicy

logger = logging.getLogger("lewm_robot.deploy")

MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


# ──────────────────────────────────────────────────────────────────────────────
# Hardware helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_robot(args):
    """Construct an SOFollower with two OpenCV cameras (up + side)."""
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    cameras = {
        "up": OpenCVCameraConfig(
            index_or_path=args.camera_up_index,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.fps,
        ),
        "side": OpenCVCameraConfig(
            index_or_path=args.camera_side_index,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.fps,
        ),
    }

    robot_cfg_kwargs = dict(port=args.port, cameras=cameras)
    if args.max_relative_target is not None:
        robot_cfg_kwargs["max_relative_target"] = args.max_relative_target

    robot = SOFollower(SOFollowerRobotConfig(**robot_cfg_kwargs))
    robot.connect()
    logger.info("Robot connected on %s | cameras: up=%d side=%d",
                args.port, args.camera_up_index, args.camera_side_index)
    return robot


def obs_to_batch(obs: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Convert SO-100 robot observation dict → policy batch dict.

    Robot returns camera frames as (H, W, 3) uint8 numpy arrays with plain
    keys (``"up"``, ``"side"``). Converts each to (3, H, W) uint8 CHW tensor
    under the full ``observation.images.*`` key.
    """
    batch = {}
    for cam_name, policy_key in [
        ("up",   "observation.images.up"),
        ("side", "observation.images.side"),
    ]:
        frame = obs.get(cam_name)
        if frame is None:
            continue
        if isinstance(frame, np.ndarray):
            chw = torch.from_numpy(frame.copy()).permute(2, 0, 1).contiguous()
        else:
            chw = frame.permute(2, 0, 1).contiguous() if frame.ndim == 3 else frame
        if chw.dtype != torch.uint8:
            chw = (chw.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        batch[policy_key] = chw  # keep on CPU — policy moves to device internally
    return batch


# ──────────────────────────────────────────────────────────────────────────────
# Goal capture
# ──────────────────────────────────────────────────────────────────────────────

def capture_goal_interactive(robot, save_path: Path, image_keys: list[str]) -> dict[str, torch.Tensor]:
    """Move robot to goal pose manually, then press Enter to capture."""
    print("\n[Goal capture] Move the arm to the desired GOAL configuration.")
    input("  Press Enter to capture goal image(s)... ")

    obs = robot.get_observation()
    goal_images = {}
    cam_map = {"observation.images.up": "up", "observation.images.side": "side"}
    for policy_key in image_keys:
        cam_name = cam_map.get(policy_key, policy_key.split(".")[-1])
        frame = obs[cam_name]
        chw = torch.from_numpy(frame.copy()).permute(2, 0, 1).contiguous()
        goal_images[policy_key] = chw
        img_path = save_path.parent / f"goal_{cam_name}.jpg"
        Image.fromarray(frame).save(img_path)
        logger.info("Saved goal image: %s", img_path)

    print("[Goal capture] Done. Starting inference loop...\n")
    return goal_images


def load_goal_from_file(goal_path: Path, keys: list[str]) -> dict[str, torch.Tensor]:
    """Load a single goal image and replicate it for all camera keys."""
    img = Image.open(goal_path).convert("RGB")
    chw = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).contiguous()
    return {k: chw for k in keys}


# ──────────────────────────────────────────────────────────────────────────────
# Dry-run dataset replay backend
# ──────────────────────────────────────────────────────────────────────────────

class DatasetReplay:
    """Serves observations from a LeRobotDataset instead of real hardware."""

    def __init__(self, repo_id: str, root: str | None, max_steps: int):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        root_path = Path(root).expanduser() if root else None
        self._ds = LeRobotDataset(repo_id=repo_id, root=root_path)
        self._idx = 0
        self._max = min(max_steps, len(self._ds))
        logger.info("Dry-run: replaying %d steps from %s", self._max, repo_id)

    def get_observation(self) -> dict:
        item = self._ds[min(self._idx, self._max - 1)]
        obs: dict = {}
        for k in ("observation.images.up", "observation.images.side"):
            if k in item:
                frame = item[k]
                # Real hardware returns (H, W, C) uint8 numpy — match that interface.
                if torch.is_tensor(frame):
                    if frame.ndim == 3 and frame.shape[0] == 3:  # CHW → HWC numpy
                        frame = (frame.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                    elif frame.ndim == 3 and frame.shape[-1] == 3:  # already HWC
                        frame = (frame.numpy() * 255).clip(0, 255).astype(np.uint8)
                obs[k.split(".")[-1]] = frame
        state = item.get("observation.state")
        if state is not None:
            for i, m in enumerate(MOTOR_NAMES):
                obs[f"{m}.pos"] = float(state[i])
        self._idx = min(self._idx + 1, self._max)
        return obs

    def send_action(self, action: dict) -> dict:
        return action

    def disconnect(self):
        logger.info("Dry-run finished after %d steps.", self._idx)


# ──────────────────────────────────────────────────────────────────────────────
# Main control loop
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def deploy(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    logger.info("Device: %s", device)

    # ── Build policy ────────────────────────────────────────────────────────
    config = JEPAConfig(
        world_model_path=args.world_model_path,
        gc_idm_path=args.gc_idm_path,
        normalizers_path=args.normalizers_path,
        image_keys=args.image_keys,
        encoder_scale=args.encoder_scale,
        embed_dim=args.embed_dim,
        history_size=args.history_size,
        action_dim=len(MOTOR_NAMES),
        frameskip=args.frameskip,
        max_horizon=args.max_horizon,
        chunk_size=1,           # per-step closed-loop
        device=str(device),
    )
    logger.info("Building JEPAPolicy…")
    policy = JEPAPolicy(config)
    policy.eval()

    # ── Robot / replay backend ──────────────────────────────────────────────
    if args.dry_run_replay_from:
        robot = DatasetReplay(
            args.dry_run_replay_from,
            args.dry_run_replay_root,
            args.max_steps,
        )
    else:
        robot = build_robot(args)

    try:
        # ── Set goal ────────────────────────────────────────────────────────
        if args.capture_goal and not args.dry_run_replay_from:
            goal_save = Path(args.goal_image) if args.goal_image else Path("goal.jpg")
            goal_images = capture_goal_interactive(robot, goal_save, config.image_keys)
        elif args.goal_image:
            goal_images = load_goal_from_file(
                Path(args.goal_image).expanduser(), config.image_keys
            )
        else:
            raise ValueError(
                "Provide --goal-image <path> or use --capture-goal to snap "
                "the goal from the live cameras."
            )

        logger.info("Setting goal embedding…")
        policy.set_goal(goal_images)
        logger.info("Goal set. Starting control loop (max %d steps @ %.0f Hz).",
                    args.max_steps, args.fps)

        input("\n[Inference] Move arm to START position, then press Enter to begin.\n"
              ) if not args.dry_run_replay_from else None

        # ── Control loop ────────────────────────────────────────────────────
        control_dt = 1.0 / args.fps
        latencies: list[float] = []

        for step in range(args.max_steps):
            t0 = time.perf_counter()

            obs = robot.get_observation()
            batch = obs_to_batch(obs, device)

            # GC-IDM: single encode + MLP pass → 6-DOF action
            action_tensor = policy.select_action(batch)[0]      # (6,)
            action_dict = {
                f"{m}.pos": float(action_tensor[i])
                for i, m in enumerate(MOTOR_NAMES)
            }
            robot.send_action(action_dict)

            dt = time.perf_counter() - t0
            latencies.append(dt * 1e3)

            sleep = control_dt - dt
            if sleep > 0:
                time.sleep(sleep)
            elif step % 30 == 0 and step > 0:
                logger.warning(
                    "step=%d: loop overrun %.1f ms (budget %.1f ms)",
                    step, dt * 1e3, control_dt * 1e3,
                )

            if step % 30 == 0:
                avg_ms = sum(latencies[-30:]) / min(len(latencies), 30)
                logger.info(
                    "step=%4d | horizon=%3d | avg_inference=%.1f ms | "
                    "action=%s",
                    step, policy._horizon, avg_ms,
                    [f"{v:.2f}" for v in action_tensor.tolist()],
                )

    finally:
        if hasattr(robot, "disconnect"):
            robot.disconnect()
        if latencies:
            avg = sum(latencies) / len(latencies)
            p95 = sorted(latencies)[int(0.95 * len(latencies))]
            logger.info(
                "Session complete — %d steps | avg=%.1f ms | p95=%.1f ms",
                len(latencies), avg, p95,
            )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GC-IDM closed-loop deployment on dual-camera SO-100.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Checkpoints
    grp = p.add_argument_group("Checkpoints")
    grp.add_argument("--world-model-path", required=True,
                     help="Pickled JEPA object from train_lewm.py (*_object.ckpt)")
    grp.add_argument("--gc-idm-path", required=True,
                     help="GC-IDM state-dict from train_gc_idm.py (gc_idm.pt)")
    grp.add_argument("--normalizers-path", default=None,
                     help="Normalizer stats (*_normalizers.pt)")

    # Goal
    grp2 = p.add_argument_group("Goal")
    grp2.add_argument("--goal-image", default=None,
                      help="Path to a PNG/JPG goal image. Also used as save path for --capture-goal.")
    grp2.add_argument("--capture-goal", action="store_true",
                      help="Snap goal from live cameras (Move arm → Enter)")

    # Hardware
    grp3 = p.add_argument_group("Hardware")
    grp3.add_argument("--port", default="/dev/ttyACM0")
    grp3.add_argument("--camera-up-index", type=int, default=0,
                      help="OpenCV device index for the UP (top-down) camera")
    grp3.add_argument("--camera-side-index", type=int, default=2,
                      help="OpenCV device index for the SIDE camera")
    grp3.add_argument("--camera-width", type=int, default=640)
    grp3.add_argument("--camera-height", type=int, default=480)
    grp3.add_argument("--max-relative-target", type=float, default=None,
                      help="Per-motor safety cap (degrees). Unset = robot default.")

    # Architecture (must match Stage 1 training)
    grp4 = p.add_argument_group("Architecture")
    grp4.add_argument("--encoder-scale", default="tiny")
    grp4.add_argument("--embed-dim", type=int, default=192)
    grp4.add_argument("--history-size", type=int, default=3)
    grp4.add_argument("--frameskip", type=int, default=5)
    grp4.add_argument("--max-horizon", type=int, default=50)
    grp4.add_argument(
        "--image-keys", nargs="+",
        default=["observation.images.up"],
        help="Camera keys to use (must match Stage 1 training). "
             "Single-cam: observation.images.up  "
             "Dual-cam: observation.images.up observation.images.side",
    )

    # Loop
    grp5 = p.add_argument_group("Control loop")
    grp5.add_argument("--fps", type=float, default=30.0)
    grp5.add_argument("--max-steps", type=int, default=300)
    grp5.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Dry run
    grp6 = p.add_argument_group("Dry run")
    grp6.add_argument("--dry-run-replay-from", default=None,
                      help="LeRobot dataset repo_id for hardware-free testing")
    grp6.add_argument("--dry-run-replay-root", default=None,
                      help="Local root directory for the replay dataset")

    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = build_parser().parse_args()
    deploy(args)


if __name__ == "__main__":
    main()
