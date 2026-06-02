"""Register the JEPAPolicy plugin and forward to ``lerobot-rollout``.

This wrapper ensures ``lewm_robot`` is imported (registering ``JEPAConfig``
with draccus) before the lerobot rollout machinery starts, so
``--policy.type=jepa`` resolves correctly without patching lerobot source.

Usage::

    python -m lewm_robot.rollout_jepa \\
        --policy.type=jepa \\
        --policy.world_model_path=~/.stable_worldmodel/<run>/lewm_so100_epoch_100_object.ckpt \\
        --policy.gc_idm_path=~/.stable_worldmodel/<run>/gc_idm.pt \\
        --policy.normalizers_path=~/.stable_worldmodel/<run>/lewm_so100_normalizers.pt \\
        --policy.goal_image_path=./goal.jpg \\
        --policy.image_keys='["observation.images.up","observation.images.side"]' \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyACM0 \\
        --robot.cameras='{"up":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"side":{"type":"opencv","index_or_path":2,"width":640,"height":480,"fps":30}}' \\
        --duration=60

Notes
-----
- Camera keys in ``--robot.cameras`` must be plain names (``up``, ``side``), matching
  the last component of the ``image_keys`` entries (``observation.images.up`` → ``up``).
- ``lerobot-rollout`` calls ``policy.reset()`` between episodes; the goal embedding
  is re-loaded from ``goal_image_path`` automatically in ``JEPAPolicy.__init__``
  and stays fixed for the whole session. To change the goal, restart with a
  different ``--policy.goal_image_path``.
- For interactive goal capture (press Enter to snap goal from the live cameras)
  use ``lewm_robot.deploy_jepa_so100`` instead.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import lewm_robot  # noqa: F401, E402  — registers JEPAConfig with draccus

from lerobot.scripts.lerobot_rollout import main  # noqa: E402


if __name__ == "__main__":
    main()
