"""Pre/post-processor for the world-model planning policy.

The policy already handles its own image preprocessing and action scaling, so
the LeRobot processor pipeline is a pass-through. This module exists so the
factory route in ``lerobot.policies.factory.make_pre_post_processors`` can
locate it (the dynamic-import path looks for
``lewm.policies.wm_planning.processor_wm_planning.make_wm_planning_pre_post_processors``).
"""

from __future__ import annotations

from lerobot.processor import (
    PolicyProcessorPipeline,
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)

from lewm_robot.policies.wm_planning.configuration_wm_planning import WMPlanningConfig


def make_wm_planning_pre_post_processors(
    config: WMPlanningConfig,
    dataset_stats: dict | None = None,
) -> tuple[
    PolicyProcessorPipeline,
    PolicyProcessorPipeline,
]:
    """Return identity pre/post processor pipelines."""
    del dataset_stats  # unused — policy handles its own normalization

    preprocessor: PolicyProcessorPipeline = PolicyProcessorPipeline(
        steps=[],
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor: PolicyProcessorPipeline = PolicyProcessorPipeline(
        steps=[],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor
