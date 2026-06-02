"""Pre/post-processor pipeline for JEPAPolicy.

The policy handles its own image normalisation and action denormalisation
internally (matching the pattern in lewm.policies.wm_planning), so the
LeRobot processor pipeline is a pass-through.

This module exists so ``lerobot.policies.factory._make_processors_from_policy_config``
can locate it via the dynamic import path derived from the config class module:
    configuration_jepa  →  processor_jepa
    function: make_jepa_pre_post_processors
"""

from __future__ import annotations

from lerobot.processor import (
    PolicyProcessorPipeline,
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)

from lewm_robot.policies.jepa.configuration_jepa import JEPAConfig


def make_jepa_pre_post_processors(
    config: JEPAConfig,
    dataset_stats: dict | None = None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Return identity pre/post processor pipelines for JEPAPolicy.

    JEPAPolicy performs its own preprocessing (ImageNet normalisation,
    resize) inside ``select_action``, so no transforms are applied here.
    The ``dataset_stats`` argument is accepted for API compatibility but unused.
    """
    del config, dataset_stats

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
