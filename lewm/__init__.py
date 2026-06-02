"""lewm — Latent World Model package.

Re-exports the JEPA model class so that pickled ``_object.ckpt`` files
(saved by ``utils.ModelObjectCallBack``) can be deserialized after
``import lewm``. The deploy script and policy plugin both rely on this.
"""

# The JEPA module lives at the repo root (jepa.py); we re-export here so
# downstream code can do `from lewm import JEPA` without depending on the
# repo root being on sys.path.
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from jepa import JEPA  # noqa: E402

__all__ = ["JEPA"]
