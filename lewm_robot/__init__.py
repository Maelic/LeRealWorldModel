"""lewm_robot — JEPA world model + GC-IDM amortized planner as a LeRobot policy.

Importing this package registers ``JEPAConfig`` in LeRobot's draccus
ChoiceRegistry so that ``lerobot.policies.factory`` can discover it via the
``--discover_packages_path lewm_robot`` flag without any modifications to
LeRobot's source code.
"""

from lewm_robot.policies.jepa.configuration_jepa import JEPAConfig  # noqa: F401
