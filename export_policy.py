"""Export a trained JEPA + GC-IDM policy as a LeRobot-compatible checkpoint.

Usage:
    python export_policy.py \\
        --world_model_path ~/.stable_worldmodel/<run_id>/lewm_so100_epoch_100_object.ckpt \\
        --gc_idm_path      ~/.stable_worldmodel/<run_id>/gc_idm.pt \\
        --normalizers_path ~/.stable_worldmodel/<run_id>/lewm_so100_normalizers.pt \\
        --goal_image_path  /path/to/goal.jpg \\
        --output_dir       checkpoints/jepa_so100

After export, deploy with:
    python -m lerobot.scripts.control_robot \\
        --policy.path=checkpoints/jepa_so100 \\
        --robot.type=so_follower \\
        --discover_packages_path=lewm_robot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import lewm_robot  # noqa: F401 — triggers JEPAConfig registration

from lewm_robot.policies.jepa.configuration_jepa import JEPAConfig
from lewm_robot.policies.jepa.modeling_jepa import JEPAPolicy
from lewm_robot.policies.jepa.processor_jepa import make_jepa_pre_post_processors


def main() -> None:
    parser = argparse.ArgumentParser(description="Export JEPAPolicy checkpoint.")
    parser.add_argument("--world_model_path", required=True)
    parser.add_argument("--gc_idm_path", required=True)
    parser.add_argument("--normalizers_path", default=None)
    parser.add_argument("--goal_image_path", default=None)
    parser.add_argument("--output_dir", default="checkpoints/jepa_so100")
    # Architecture (should match training)
    parser.add_argument("--encoder_scale", default="tiny")
    parser.add_argument("--embed_dim", type=int, default=192)
    parser.add_argument("--history_size", type=int, default=3)
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--max_horizon", type=int, default=50)
    parser.add_argument(
        "--image_keys",
        nargs="+",
        default=["observation.images.up", "observation.images.side"],
    )
    parser.add_argument("--chunk_size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = JEPAConfig(
        world_model_path=args.world_model_path,
        gc_idm_path=args.gc_idm_path,
        normalizers_path=args.normalizers_path,
        goal_image_path=args.goal_image_path,
        image_keys=args.image_keys,
        encoder_scale=args.encoder_scale,
        embed_dim=args.embed_dim,
        history_size=args.history_size,
        action_dim=args.action_dim,
        frameskip=args.frameskip,
        max_horizon=args.max_horizon,
        chunk_size=args.chunk_size,
        device=args.device,
    )

    print("Building JEPAPolicy…")
    policy = JEPAPolicy(config)
    policy.eval()

    print("Saving policy to", output_dir)
    policy.save_pretrained(output_dir)

    # Save processor pipelines so LeRobot can load them from the checkpoint dir
    preprocessor, postprocessor = make_jepa_pre_post_processors(config)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)

    # Also copy normalizer stats into the checkpoint dir for portability
    if args.normalizers_path:
        norm_src = Path(args.normalizers_path).expanduser()
        if norm_src.exists():
            import shutil
            shutil.copy(norm_src, output_dir / "normalizers.pt")

    print("\nCheckpoint contents:")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size
        print(f"  {f.name:40s}  {size:>10,} bytes")

    print(f"\nDeploy with:\n"
          f"  python -m lerobot.scripts.control_robot \\\n"
          f"    --policy.path={output_dir} \\\n"
          f"    --robot.type=so_follower \\\n"
          f"    --discover_packages_path=lewm_robot")


if __name__ == "__main__":
    main()
