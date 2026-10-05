"""jevgate — a calibrated tool-call safety gate.

One call is the whole product:

    from jevgate import gate

    verdict = gate("bash", {"command": "rm -rf build_cache"})
    if verdict.verdict == "block":
        ...
"""

from .client import Decision, JevClient, JevError
from .gate import GATE_QUESTIONS, Gate, ToolCall, VerdictResult, match_bypass
from .guard import CommandBlocked, GuardResult, guarded

__version__ = "0.2.0"

_DEFAULT_GATE: Gate | None = None


def gate(
    tool: str,
    arguments: dict,
    *,
    mode: str = "shadow",
    threshold_confirm: float = 0.15,
    threshold_block: float = 0.60,
    bypass_patterns: list[str] | None = None,
    client: JevClient | None = None,
    **kwargs,
) -> VerdictResult:
    """Judge one tool call and return allow / confirm / block.

    Args:
        tool: tool name, e.g. "bash", "write", "edit".
        arguments: the tool's arguments as a dict.
        mode: "shadow" (judge, block nothing), "enforce" (return the verdict
              for real), or "off".
        threshold_confirm / threshold_block: risk levels for each verdict.
        bypass_patterns: glob patterns the user has approved, e.g.
              ["npm test*", "git status"]. A match returns allow with no API
              call; check `verdict.bypassed_by` to tell that apart from a
              judged allow. Ignored in shadow mode.
        client: a JevClient. Defaults to one reading TYPESAFE_API_KEY. Mostly
                useful for tests, which pass a stub.

    Never raises: an unreachable API yields allow with an error field set.
    """
    global _DEFAULT_GATE
    config = (
        mode,
        threshold_confirm,
        threshold_block,
        tuple(bypass_patterns or ()),
        id(client),
    )
    if _DEFAULT_GATE is None or _DEFAULT_GATE._config != config:
        _DEFAULT_GATE = Gate(
            client=client,
            mode=mode,  # type: ignore[arg-type]
            threshold_confirm=threshold_confirm,
            threshold_block=threshold_block,
            bypass_patterns=bypass_patterns,
        )
    return _DEFAULT_GATE.check_tool(tool, arguments, **kwargs)


__all__ = [
    "gate",
    "Gate",
    "ToolCall",
    "VerdictResult",
    "JevClient",
    "Decision",
    "JevError",
    "GATE_QUESTIONS",
    "match_bypass",
    "guarded",
    "GuardResult",
    "CommandBlocked",
    "__version__",
]
