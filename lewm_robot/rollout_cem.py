"""Thin wrapper that registers the wm_planning policy plugin and then forwards
to ``lerobot-rollout`` so users can run it without packaging lewm as a
``lerobot_policy_*`` distribution.

Usage::

    python -m lewm.rollout_wm_planning \
        --strategy.type=base \
        --policy.type=wm_planning \
        --policy.world_model_path=~/.stable_worldmodel/<run_id>/lewm_epoch_100_object.ckpt \
        --policy.normalizers_path=~/.stable_worldmodel/<run_id>/lewm_normalizers.pt \
        --policy.goal_image_path=./goal.png \
        --robot.type=so100_follower \
        --robot.port=/dev/ttyACM0 \
        --robot.cameras='{"front":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30}}' \
        --task="pick up cube" --duration=60

Once imported, ``lewm.policies.wm_planning`` registers
``WMPlanningConfig`` with draccus' choice registry, so the
``--policy.type=wm_planning`` CLI flag resolves correctly.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import lewm_robot.policies.wm_planning  # noqa: F401, E402  (registers the policy)

from lerobot.scripts.lerobot_rollout import main  # noqa: E402


if __name__ == "__main__":
    main()
