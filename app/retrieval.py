from __future__ import annotations

import math
import re
from collections import Counter

from .knowledge import KnowledgeBase
from .models import Runbook, SearchResult


def tokenize(text: str) -> list[str]:
    """Small dependency-free tokenizer for the synthetic Chinese/English corpus."""
    normalized = text.casefold()
    tokens = re.findall(r"[a-z0-9]+", normalized)
    for sequence in re.findall(r"[\u4e00-\u9fff]+", normalized):
        if len(sequence) == 1:
            tokens.append(sequence)
        else:
            tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tokens


def _snippet(document: Runbook, query_tokens: set[str], limit: int = 110) -> str:
    candidates = [
        line.strip("- ")
        for line in document.body.splitlines()
        if line.strip().startswith("-")
    ]
    for candidate in candidates:
        if query_tokens.intersection(tokenize(candidate)):
            return candidate[:limit]
    return (candidates[0] if candidates else document.body.replace("\n", " "))[:limit]


class KeywordRetriever:
    """BM25-style lexical baseline implemented with the standard library."""

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self.kb = knowledge_base
        self._documents = knowledge_base.documents
        self._tokens = {
            document.id: tokenize(f"{document.title} {document.service} {document.body}")
            for document in self._documents
        }
        self._document_frequency: Counter[str] = Counter()
        for tokens in self._tokens.values():
            self._document_frequency.update(set(tokens))
        self._average_length = sum(map(len, self._tokens.values())) / len(self._tokens)

    def search(self, query: str, limit: int = 3) -> list[SearchResult]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        counts = Counter(query_tokens)
        scored: list[SearchResult] = []
        for document in self._documents:
            score, matched = self._score(document.id, counts)
            if score <= 0:
                continue
            scored.append(
                SearchResult(
                    id=document.id,
                    title=document.title,
                    service=document.service,
                    score=score,
                    snippet=_snippet(document, set(query_tokens)),
                    reasons=tuple(f"词面命中：{token}" for token in matched[:4]),
                )
            )
        return sorted(scored, key=lambda item: (-item.score, item.id))[:limit]

    def _score(self, document_id: str, query_counts: Counter[str]) -> tuple[float, list[str]]:
        tokens = self._tokens[document_id]
        frequencies = Counter(tokens)
        length = len(tokens)
        number_of_documents = len(self._documents)
        score = 0.0
        matched: list[str] = []
        k1, b = 1.5, 0.75
        for token, query_frequency in query_counts.items():
            term_frequency = frequencies[token]
            if not term_frequency:
                continue
            matched.append(token)
            document_frequency = self._document_frequency[token]
            inverse_frequency = math.log(
                1 + (number_of_documents - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = term_frequency + k1 * (
                1 - b + b * length / max(self._average_length, 1)
            )
            score += query_frequency * inverse_frequency * term_frequency * (k1 + 1) / denominator
        return score, matched


class SemanticRetriever:
    """
    Transparent semantic baseline.

    It maps domain paraphrases to a maintained concept vector and combines that
    score with a small lexical signal. This is intentionally lightweight and
    reproducible offline; it is not presented as a general-purpose embedding model.
    """

    def __init__(self, knowledge_base: KnowledgeBase, keyword: KeywordRetriever) -> None:
        self.kb = knowledge_base
        self.keyword = keyword

    def search(self, query: str, limit: int = 3) -> list[SearchResult]:
        query_concepts = set(self.kb.detect_concepts(query))
        lexical_results = self.keyword.search(query, limit=len(self.kb.documents))
        lexical_scores = {item.id: item.score for item in lexical_results}
        max_lexical = max(lexical_scores.values(), default=0.0)

        scored: list[SearchResult] = []
        for document in self.kb.documents:
            overlap = query_concepts.intersection(document.concepts)
            concept_score = len(overlap) / max(len(query_concepts), 1)
            lexical_score = lexical_scores.get(document.id, 0.0) / max(max_lexical, 1.0)
            score = 0.85 * concept_score + 0.15 * lexical_score
            if not query_concepts:
                score = 0.45 * lexical_score
            if score <= 0:
                continue
            reasons = [f"语义概念：{self.kb.concept_label(item)}" for item in sorted(overlap)]
            if lexical_score:
                reasons.append("辅以词面相关性")
            scored.append(
                SearchResult(
                    id=document.id,
                    title=document.title,
                    service=document.service,
                    score=score,
                    snippet=_snippet(document, set(tokenize(query))),
                    reasons=tuple(reasons),
                )
            )
        return sorted(scored, key=lambda item: (-item.score, item.id))[:limit]
