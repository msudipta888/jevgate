"""Turn a shadow-mode log into measured numbers.

This is the part that makes the project defensible. You run the gate in shadow
mode, hand-label the logged calls, then compute precision/recall/ROC over the
gate's p_destructive so the threshold is derived from data rather than guessed.

No numpy/sklearn: the arithmetic is a few lines and keeps the dependency list
at zero, which matters for something meant to be dropped into other tools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass
class LabeledCall:
    """One logged call plus the human label for whether it was truly dangerous."""

    fingerprint: str
    tool: str
    arguments: dict[str, Any]
    p_destructive: float
    p_exfiltration: float
    p_privilege: float
    label: int  # 1 = genuinely dangerous, 0 = genuinely safe
    note: str = ""
    latency_ms: float = float("nan")


@dataclass
class Report:
    n_labeled: int
    n_positive: int
    threshold: float
    precision: float
    recall: float
    f1: float
    specificity: float
    blocked: int
    false_positives: int
    false_negatives: int
    brier: float
    curve: list[tuple[float, float, float]] = field(default_factory=list)  # (threshold, precision, recall)
    mean_latency_ms: float = float("nan")

    def render(self) -> str:
        latency = "n/a (no latency in labels)" if self.mean_latency_ms != self.mean_latency_ms else f"{self.mean_latency_ms:.0f} ms"
        lines = [
            f"labeled calls      : {self.n_labeled}",
            f"truly dangerous     : {self.n_positive}",
            f"mean gate latency   : {latency}",
            f"calibration (Brier) : {self.brier:.4f}  (lower is better; 0.25 = coin flip)",
            "",
            f"threshold           : {self.threshold:.3f}",
            f"precision           : {self.precision:.3f}  (of flagged, how many were real)",
            f"recall              : {self.recall:.3f}  (of real dangers, how many we caught)",
            f"f1                  : {self.f1:.3f}",
            f"specificity         : {self.specificity:.3f}",
            f"flagged             : {self.blocked}",
            f"false positives     : {self.false_positives}",
            f"false negatives     : {self.false_negatives}",
        ]
        return "\n".join(lines)

    def csv(self) -> str:
        out = ["threshold,precision,recall"]
        out += [f"{t:.3f},{p:.4f},{r:.4f}" for t, p, r in self.curve]
        return "\n".join(out)


def load_labels(path: str | Path) -> list[LabeledCall]:
    """Read a labels file: JSONL rows with at least the risk numbers and a label.

    Accepts a hand-written file of JSONL rows carrying at least the three risk
    numbers and a "label": 1 for genuinely dangerous, 0 for genuinely safe.
    Read one in yourself from wherever your own traffic is recorded — jevgate
    does not write files.
    """
    rows: list[LabeledCall] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        if "label" not in raw:
            continue
        rows.append(
            LabeledCall(
                fingerprint=raw.get("fingerprint", ""),
                tool=raw.get("tool", ""),
                arguments=raw.get("arguments", {}),
                p_destructive=float(raw.get("p_destructive", 0.0)),
                p_exfiltration=float(raw.get("p_exfiltration", 0.0)),
                p_privilege=float(raw.get("p_privilege", 0.0)),
                label=int(raw["label"]),
                note=raw.get("note", ""),
                latency_ms=float(raw.get("latency_ms", float("nan"))),
            )
        )
    return rows


def risk_score(call: LabeledCall) -> float:
    """Combine the three risk channels into the single number we threshold.

    Deliberately in code, not in the model. The max() keeps a single confident
    signal (e.g. obvious exfiltration) from being averaged away by two quiet
    ones — an attacker only has to trip one detector.
    """
    return max(call.p_destructive, call.p_exfiltration, call.p_privilege)


def _metrics(calls: Sequence[LabeledCall], threshold: float) -> dict[str, float]:
    tp = fp = fn = tn = 0
    for call in calls:
        flagged = risk_score(call) >= threshold
        if flagged and call.label == 1:
            tp += 1
        elif flagged and call.label == 0:
            fp += 1
        elif not flagged and call.label == 1:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def brier_score(calls: Sequence[LabeledCall]) -> float:
    """Mean squared error between predicted risk and the true label.

    This is the calibration check. If this sits near 0.25 the probabilities are
    noise and no threshold will be trustworthy.
    """
    if not calls:
        return float("nan")
    return sum((risk_score(c) - c.label) ** 2 for c in calls) / len(calls)


def evaluate(
    calls: Iterable[LabeledCall],
    target_precision: float = 0.95,
    curve_points: int = 50,
) -> Report:
    """Sweep thresholds and pick the lowest one that still hits target precision."""
    calls = list(calls)
    if not calls:
        raise ValueError("No labeled calls found. Run shadow mode and label the log first.")

    curve: list[tuple[float, float, float]] = []
    for i in range(curve_points + 1):
        threshold = i / curve_points
        m = _metrics(calls, threshold)
        curve.append((threshold, m["precision"], m["recall"]))

    # Among thresholds that hit the precision target, take the lowest one — it
    # catches the most real dangers. `recall > 0` stops a threshold that flags
    # nothing from "qualifying" on empty precision.
    qualifying = [
        (t, p, r) for t, p, r in curve if p >= target_precision and r > 0.0
    ]
    chosen = min(qualifying)[0] if qualifying else 0.0

    final = _metrics(calls, chosen)
    latencies = [c.latency_ms for c in calls if c.latency_ms == c.latency_ms]

    return Report(
        n_labeled=len(calls),
        n_positive=sum(c.label for c in calls),
        threshold=chosen,
        precision=final["precision"],
        recall=final["recall"],
        f1=final["f1"],
        specificity=final["specificity"],
        blocked=int(final["tp"] + final["fp"]),
        false_positives=int(final["fp"]),
        false_negatives=int(final["fn"]),
        brier=brier_score(calls),
        mean_latency_ms=(sum(latencies) / len(latencies)) if latencies else float("nan"),
        curve=curve,
    )
