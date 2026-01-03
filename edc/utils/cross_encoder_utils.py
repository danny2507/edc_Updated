from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass
class CrossEncoderReranker:
    """Cross-encoder reranker.

    This uses `sentence_transformers.CrossEncoder` under the hood.

    We keep it in a small wrapper so the rest of the codebase doesn't have to
    depend on CrossEncoder specifics.
    """

    model_name: str

    def __post_init__(self) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, candidates: Sequence[str], top_k: int) -> List[Tuple[int, float]]:
        """Return candidate indices sorted by cross-encoder score.

        Args:
            query: Query string.
            candidates: Candidate strings.
            top_k: How many reranked results to return.

        Returns:
            List of tuples (candidate_index, score), sorted descending by score.
        """
        if top_k <= 0 or not candidates:
            return []

        pairs = [(query, c) for c in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        ranked = ranked[: min(top_k, len(ranked))]
        return [(i, float(scores[i])) for i in ranked]
