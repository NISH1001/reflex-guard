"""Multi-label guardrails on System One decision models."""

from importlib.metadata import PackageNotFoundError, version

from .chunking import InputTooLong
from .compose import AllOf, AnyOf, CascadeGuard, EnsembleGuard, Rule, Votes, Weighted
from .guard import Guard
from .impls import GlinerGuard, JevGuard, LayaGuard, OpenAIDecisionGuard, VonGuard
from .modes import Mode
from .result import CategoryResult, GuardResult

try:
    __version__ = version("reflexguard")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"

__all__ = ["AllOf", "AnyOf", "CascadeGuard", "CategoryResult", "EnsembleGuard", "GlinerGuard", "Guard", "GuardResult", "InputTooLong", "JevGuard", "LayaGuard", "Mode", "OpenAIDecisionGuard", "Rule", "VonGuard", "Votes", "Weighted", "__version__"]
