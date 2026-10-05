"""Client for the Jev Decision API, on top of the official TypeSafe SDK.

Transport, retries, and response validation are the SDK's job (`typesafe-sdk`),
so this module is only the translation layer: jevgate asks noul/choice/score
questions and wants one flat `Decision` per question carrying a single
probability in [0, 1], and that is what `ask()` returns.

Fail-open by design: the gate must never take an agent down because a network
call failed. `ask()` raises JevError on unrecoverable errors and every caller
treats that as "allow".
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from typesafe_sdk import (
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    RetryPolicy,
    Score,
    ScoreAnswer,
    TypeSafeClient,
    TypeSafeError,
)

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"


class JevError(RuntimeError):
    """Raised when the Decision API cannot be reached or returns an error."""


@dataclass
class Decision:
    """One typed answer from the model, normalised across question types."""

    key: str
    type: str  # "choice" | "score" | "noul"
    value: float | str | None = None
    confidence: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def probability(self) -> float:
        """The decision's headline number as a probability in [0, 1].

        choice -> probability of the selected option
        score  -> probability-weighted position on the rubric, normalised
        noul   -> the noul value itself
        """
        if self.type == "noul":
            return float(self.value or 0.0)
        if self.type == "choice":
            return float(self.confidence or 0.0)
        if self.type == "score":
            probs = self.probabilities
            if probs:
                weighted = sum(float(k) * float(v) for k, v in probs.items())
                total = float(max(float(k) for k in probs)) or 1.0
                return weighted / total
            return float(self.value or 0.0)
        return 0.0


class JevClient:
    """Thin wrapper over `TypeSafeClient`, translating answers to Decisions.

    Questions are passed as plain dicts (`{"type": "noul", ...}`) so callers
    keep the one small question vocabulary described in the README; the SDK
    validates them before anything goes over the wire.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = DEFAULT_MODEL,
        timeout: float = 15.0,
        max_retries: int = 2,
        max_state_chars: int = 8000,
    ) -> None:
        # A .env in the cwd is a convenience for local dev. override=False keeps
        # it from winning over a real environment variable, which is what an
        # embedding program will have set.
        load_dotenv(override=False)
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY", "")
        # The SDK appends /v1/systemone to this, so it is the API root.
        self.base_url = base_url or os.getenv("TYPESAFE_BASE_URL", DEFAULT_BASE_URL)
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_state_chars = max_state_chars
        self._client: TypeSafeClient | None = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _sdk(self) -> TypeSafeClient:
        """Build the SDK client on first use.

        Lazy because the SDK rejects a missing key at construction, and a gate
        with no key must still construct so it can fail open with a warning.
        """
        if self._client is None:
            try:
                self._client = TypeSafeClient(
                    api_key=self.api_key or None,
                    base_url=self.base_url,
                    model=self.model,
                    timeout=self.timeout,
                    retry=RetryPolicy(max_retries=self.max_retries),
                )
            except TypeSafeError as exc:
                raise JevError(str(exc)) from exc
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "JevClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def ask(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Decision]:
        """Send one state plus one or more typed questions, in a single round trip."""
        if not self.configured:
            raise JevError(
                "No API key. Set TYPESAFE_API_KEY, or pass api_key= to JevClient."
            )
        if not questions:
            raise JevError("At least one question is required.")

        try:
            response = self._sdk().system_one(
                state=state[: self.max_state_chars],
                questions=questions,
            )
        except TypeSafeError as exc:
            # The SDK already applied its retry policy; anything reaching here
            # is final.
            raise JevError(f"Jev Decision API error: {exc}") from exc

        return {key: self._decision(key, answer) for key, answer in response.answers.items()}

    @staticmethod
    def _decision(key: str, answer: Any) -> Decision:
        """One SDK answer object -> one Decision, whatever the question type."""
        if isinstance(answer, NoulAnswer):
            return Decision(
                key=key,
                type="noul",
                value=float(answer.noul),
                raw=answer.model_dump(),
            )
        if isinstance(answer, ScoreAnswer):
            return Decision(
                key=key,
                type="score",
                value=float(answer.score),
                confidence=float(answer.confidence),
                probabilities={str(k): float(v) for k, v in answer.probabilities.items()},
                raw=answer.model_dump(),
            )
        if isinstance(answer, ChoiceAnswer):
            return Decision(
                key=key,
                type="choice",
                value=answer.choice,
                confidence=float(answer.confidence),
                probabilities={str(k): float(v) for k, v in answer.probabilities.items()},
                raw=answer.model_dump(),
            )
        raise JevError(f"Unrecognised answer type for question {key!r}: {type(answer).__name__}")

    @staticmethod
    def _parse(data: dict[str, Any]) -> dict[str, Decision]:
        """Normalise a raw wire response (kept for offline parsing/tests)."""
        decisions: dict[str, Decision] = {}
        for key, answer in (data.get("answers") or {}).items():
            kind = answer.get("type", "noul")
            if kind == "noul":
                value: float | str | None = answer.get("noul")
            elif kind == "choice":
                value = answer.get("choice")
            else:
                value = answer.get("score")
            decisions[key] = Decision(
                key=key,
                type=kind,
                value=value,
                confidence=answer.get("confidence"),
                probabilities=answer.get("probabilities") or {},
                raw=answer,
            )
        return decisions


# Re-exported so callers can build typed questions without importing the SDK.
__all__ = ["Decision", "JevClient", "JevError", "Noul", "Score"]
