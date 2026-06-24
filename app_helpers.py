# app_helpers.py — Pure, dependency-free helpers used by app.py.
# Kept in a separate module so they are importable by tests without pulling
# in gradio / spaces / torch / transformers.
#
# ---- Changelog ----
# [2026-06-24] CC — Task 4 review fix (Finding 1)
# What: Extracted _access_count_for() from app.py so it is unit-testable
#       without the heavy HF/Gradio/spaces dependency chain.
# Why: app.py imports spaces+gradio (unavailable locally); pure helpers must
#      be importable independently for the test suite.
# How: Thin module, no imports beyond stdlib.
# -------------------


def _access_count_for(mod, access, max_access):
    """Return the access count to feed into classify_tier for a weight module.

    A module the sensor never observed (mod not in access) is treated
    CONSERVATIVELY as fully-hot (returns max_access) so that unobserved
    weights are NEVER counted as condensable.  Savings must not be overstated.

    A module the sensor DID observe returns its actual forward_count.
    """
    if mod in access:
        return int(access[mod])
    return int(max_access)   # unobserved → HOT, not condensable
