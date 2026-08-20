from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Runbook:
    id: str
    title: str
    service: str
    concepts: tuple[str, ...]
    body: str
    path: str


@dataclass(frozen=True)
class SearchResult:
    id: str
    title: str
    service: str
    score: float
    snippet: str
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["score"] = round(self.score, 4)
        data["reasons"] = list(self.reasons)
        return data


@dataclass(frozen=True)
class ToolEvent:
    type: str
    tool: str
    arguments: dict[str, Any]
    preview: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResponse:
    answer: str
    used_documents: list[str] = field(default_factory=list)
    trace: list[ToolEvent] = field(default_factory=list)
    boundary: bool = False
    mode: str = "offline"
    latency_ms: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "used_documents": self.used_documents,
            "trace": [event.to_dict() for event in self.trace],
            "boundary": self.boundary,
            "mode": self.mode,
            "latency_ms": round(self.latency_ms, 2),
            "usage": self.usage,
        }
