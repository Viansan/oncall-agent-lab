from __future__ import annotations

import json
import re
from pathlib import Path

from .models import Runbook


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNBOOK_DIR = PROJECT_ROOT / "data" / "runbooks"
DEFAULT_CONCEPT_FILE = PROJECT_ROOT / "data" / "concepts.json"


class KnowledgeBase:
    """Loads the synthetic runbooks and exposes only ID-based, read-only access."""

    def __init__(
        self,
        runbook_dir: Path = DEFAULT_RUNBOOK_DIR,
        concept_file: Path = DEFAULT_CONCEPT_FILE,
    ) -> None:
        self._concepts = self._load_concepts(concept_file)
        self._documents = self._load_runbooks(runbook_dir)
        if not self._documents:
            raise RuntimeError(f"No runbooks found in {runbook_dir}")

    @property
    def documents(self) -> tuple[Runbook, ...]:
        return tuple(self._documents.values())

    @property
    def document_ids(self) -> tuple[str, ...]:
        return tuple(self._documents)

    def get(self, document_id: str) -> Runbook:
        try:
            return self._documents[document_id]
        except KeyError as exc:
            raise KeyError(f"Unknown runbook ID: {document_id}") from exc

    def read_runbook(self, document_id: str) -> str:
        """Read a known runbook by logical ID; filesystem paths are never accepted."""
        return self.get(document_id).body

    def detect_concepts(self, text: str) -> tuple[str, ...]:
        normalized = text.casefold()
        detected: list[str] = []
        for concept_id, item in self._concepts.items():
            if any(alias.casefold() in normalized for alias in item["aliases"]):
                detected.append(concept_id)
        return tuple(detected)

    def concept_label(self, concept_id: str) -> str:
        item = self._concepts.get(concept_id)
        return item["label"] if item else concept_id

    @staticmethod
    def section_items(document: Runbook, heading: str) -> list[str]:
        current = False
        items: list[str] = []
        for raw_line in document.body.splitlines():
            line = raw_line.strip()
            if line == f"## {heading}":
                current = True
                continue
            if current and line.startswith("## "):
                break
            if current and line.startswith("- "):
                items.append(line[2:].strip())
        return items

    @staticmethod
    def _load_concepts(path: Path) -> dict[str, dict]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        concepts: dict[str, dict] = {}
        for item in payload.get("concepts", []):
            concept_id = str(item["id"])
            aliases = [str(alias) for alias in item.get("aliases", [])]
            if not aliases:
                continue
            concepts[concept_id] = {
                "label": str(item.get("label") or concept_id),
                "aliases": aliases,
            }
        return concepts

    @classmethod
    def _load_runbooks(cls, directory: Path) -> dict[str, Runbook]:
        documents: dict[str, Runbook] = {}
        for path in sorted(directory.glob("*.md")):
            metadata, body = cls._parse_markdown(path.read_text(encoding="utf-8"))
            document_id = metadata.get("id", "").strip()
            if not re.fullmatch(r"[a-z0-9-]+", document_id):
                raise ValueError(f"Invalid runbook ID in {path.name}")
            if document_id in documents:
                raise ValueError(f"Duplicate runbook ID: {document_id}")
            concepts = tuple(
                concept.strip()
                for concept in metadata.get("concepts", "").split(",")
                if concept.strip()
            )
            documents[document_id] = Runbook(
                id=document_id,
                title=metadata.get("title", document_id).strip(),
                service=metadata.get("service", "未命名服务").strip(),
                concepts=concepts,
                body=body.strip(),
                path=str(path.relative_to(PROJECT_ROOT)),
            )
        return documents

    @staticmethod
    def _parse_markdown(text: str) -> tuple[dict[str, str], str]:
        if not text.startswith("---\n"):
            raise ValueError("Runbook must start with front matter")
        try:
            raw_meta, body = text[4:].split("\n---\n", 1)
        except ValueError as exc:
            raise ValueError("Runbook front matter is not closed") from exc
        metadata: dict[str, str] = {}
        for line in raw_meta.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
        return metadata, body
