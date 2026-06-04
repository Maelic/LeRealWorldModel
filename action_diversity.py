"""
Action corruption utility for identifiability experiments.

Provides ``corrupt_actions``, which breaks the coupling between actions and the
state transitions they caused — while preserving the action marginal — so the
identifiability suite can measure how much a world model actually relies on its
action conditioning.

Theoretical motivation:
    Nonlinear ICA shows identifiability requires "sufficient variability" in the
    auxiliary variable (actions). Severing the action->transition pairing lets us
    measure identifiability as a function of that coupling: sweep the corruption
    level and watch the equivariance / action-dependence metrics degrade.
"""

import numpy as np


def corrupt_actions(
    actions: np.ndarray,
    corruption_level: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Corrupt actions to reduce action-state coupling (breaks identifiability).

    At corruption_level=0, actions are unchanged (maximally informative).
    At corruption_level=1, actions are fully shuffled (non-informative).

    This directly tests the theoretical claim: without action conditioning,
    identifiability degrades to arbitrary diffeomorphisms.

    Args:
        actions: (B, T, A) or (N, A) action array
        corruption_level: float in [0, 1]
        seed: random seed

    Returns:
        corrupted actions (same shape)
    """
    rng = np.random.default_rng(seed)
    actions = actions.copy()
    shape = actions.shape

    flat = actions.reshape(-1, shape[-1])
    N = len(flat)

    n_corrupt = int(N * corruption_level)
    if n_corrupt > 0:
        corrupt_idx = rng.choice(N, n_corrupt, replace=False)
        shuffled_idx = rng.permutation(corrupt_idx)
        flat[corrupt_idx] = flat[shuffled_idx]

    return flat.reshape(shape)
