from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    # Simple whitespace tokenization + lowercasing.
    # (Good enough because we're matching short relation names.)
    return (text or "").lower().split()


@dataclass
class BM25Index:
    """A tiny BM25 wrapper for prefiltering relation candidates."""

    documents: Sequence[str]
    _bm25: BM25Okapi

    @classmethod
    def build(cls, documents: Sequence[str]) -> "BM25Index":
        tokenized = [_tokenize(d) for d in documents]
        return cls(documents=documents, _bm25=BM25Okapi(tokenized))

    def topk(self, query: str, k: int) -> List[int]:
        if k <= 0:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        # argsort descending without numpy dependency here
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[: min(k, len(ranked))]
