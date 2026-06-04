"""WMPlanningPolicy — JEPA world model + sampling-based MPC, exposed as a
LeRobot ``PreTrainedPolicy``.

Importing the configuration module here registers the policy with
``PreTrainedConfig`` so ``lerobot-rollout --policy.type=wm_planning`` can
resolve it.
"""

from lewm_robot.policies.wm_planning.configuration_wm_planning import WMPlanningConfig  # noqa: F401

__all__ = ["WMPlanningConfig"]
