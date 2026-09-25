"""Multi-label guardrails on System One decision models."""

from importlib.metadata import PackageNotFoundError, version

from .guard import Guard
from .impls import GlinerGuard, JevGuard, LayaGuard, VonGuard
from .modes import Mode
from .result import CategoryResult, GuardResult

try:
    __version__ = version("reflexguard")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"

__all__ = ["CategoryResult", "GlinerGuard", "Guard", "GuardResult", "JevGuard", "LayaGuard", "Mode", "VonGuard", "__version__"]
