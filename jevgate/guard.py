"""Guarded execution for code that runs commands itself.

Any code that shells out can route through `guarded`, which judges the command
and then runs it — or refuses, or asks — before anything happens.

    from jevgate.guard import guarded

    guarded(["pytest", "-q"])                    # allowed -> runs
    guarded(["rm", "-rf", "node_modules"])       # risky   -> raises or prompts

Nothing can skip this, because there is no model in the loop deciding whether
to call it.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .gate import Gate, Mode, ToolCall, VerdictResult


class CommandBlocked(RuntimeError):
    """Raised when the gate refuses a command and the policy is `block`."""

    def __init__(self, verdict: VerdictResult, command: Sequence[str]) -> None:
        self.verdict = verdict
        self.command = list(command)
        detail = (
            f"jevgate blocked: {' '.join(self.command)}\n"
            f"  p_destructive={verdict.p_destructive:.3f} "
            f"p_exfiltration={verdict.p_exfiltration:.3f} "
            f"p_privilege={verdict.p_privilege:.3f}"
        )
        super().__init__(detail)


@dataclass
class GuardResult:
    """What happened to the command, whether or not it ran."""

    verdict: str
    allowed: bool
    executed: bool
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""

    def __bool__(self) -> bool:
        return self.allowed


def guarded(
    command: Sequence[str] | str,
    *,
    gate: Gate | None = None,
    cwd: str | None = None,
    on_confirm: Callable[[VerdictResult], bool] | None = None,
    check_only: bool = False,
    raise_on_block: bool = True,
    capture_output: bool = True,
    bypass_patterns: Sequence[str] | None = None,
    **kwargs: Any,
) -> GuardResult:
    """Judge a command, then run it if the verdict permits.

    Args:
        command: argv list, or a shell string (passed to your shell, so prefer
                 the list form).
        gate: a configured Gate; one is created with `enforce` if omitted.
        on_confirm: called for a `confirm` verdict. Return True to proceed.
                     Default is to refuse.
        check_only: judge the command but never execute it.
        raise_on_block: raise CommandBlocked on `block` (default), or return a
                        GuardResult with allowed=False. Use False with
                        check_only to inspect a verdict without a try/except —
                        that is the dry-run / CI-audit shape.
        capture_output: capture stdout/stderr into the result.
        bypass_patterns: glob patterns to allow outright with no API call, e.g.
                        ["npm test*"]. Ignored if `gate` is supplied, since the
                        gate already holds its own allowlist.

    Raises:
        CommandBlocked: on `block` (unless raise_on_block=False), or on
                        `confirm` with no approving `on_confirm`.
    """
    gate = gate or Gate(mode="enforce", bypass_patterns=bypass_patterns)
    argv = [command] if isinstance(command, str) else list(command)
    tool = "shell" if isinstance(command, str) else "exec"

    result = gate.judge(ToolCall(tool=tool, arguments={"command": command}, cwd=cwd))

    if result.verdict == "block" and raise_on_block:
        raise CommandBlocked(result, argv)

    if result.verdict == "block":
        return GuardResult(verdict="block", allowed=False, executed=False)

    if result.verdict == "confirm":
        approved = bool(on_confirm(result)) if on_confirm else False
        if not approved and raise_on_block:
            raise CommandBlocked(result, argv)
        if not approved:
            return GuardResult(verdict="confirm", allowed=False, executed=False)

    if check_only:
        return GuardResult(verdict=result.verdict, allowed=True, executed=False)

    completed = subprocess.run(  # noqa: S603 - this is the point of the function
        command,
        shell=isinstance(command, str),
        cwd=cwd,
        capture_output=capture_output,
        text=True,
        **kwargs,
    )
    return GuardResult(
        verdict=result.verdict,
        allowed=True,
        executed=True,
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )