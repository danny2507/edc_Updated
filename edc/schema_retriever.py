from typing import List, Optional

import numpy as np

import edc.utils.llm_utils as llm_utils
from edc.utils.bm25_utils import BM25Index
from edc.utils.cross_encoder_utils import CrossEncoderReranker
from edc.utils.relation_text import relation_to_text


class SchemaRetriever:
    # The class to handle the last stage: Schema Canonicalization
    def __init__(
        self,
        target_schema_dict: dict,
        embedding_model,
        embedding_tokenizer,
        finetuned_e5mistral: bool = False,
        bm25_top_k: int = 200,
        use_bm25: bool = True,
        cross_encoder_model_name: Optional[str] = None,
        cross_top_k: int = 20,
    ) -> None:
        # The canonicalizer uses an embedding model to first fetch candidates from the target schema, then uses a verifier schema to decide which one to canonicalize to or not
        # canonoicalize at all.

        self.target_schema_dict = target_schema_dict
        self.embedding_model = embedding_model
        self.embedding_tokenizer = embedding_tokenizer
        self.bm25_top_k = bm25_top_k
        self.use_bm25 = use_bm25
        self.cross_top_k = cross_top_k

        self.cross_encoder = (
            CrossEncoderReranker(cross_encoder_model_name) if cross_encoder_model_name else None
        )

        # Embed the target schema

        self.target_schema_embedding_dict = {}
        self.finetuned_e5mistral = finetuned_e5mistral

        # Precompute BM25 over relation names (lexical filter).
        self._relation_list = list(target_schema_dict.keys())
        self._bm25_index = BM25Index.build(self._relation_list) if self.use_bm25 else None

        # Bi-encoder embeddings over (name + definition)
        for relation, relation_definition in target_schema_dict.items():
            relation_text = relation_to_text(relation, relation_definition)
            if finetuned_e5mistral:
                embedding = llm_utils.get_embedding_e5mistral(
                    self.embedding_model,
                    self.embedding_tokenizer,
                    relation_text,
                )
            else:
                embedding = llm_utils.get_embedding_sts(
                    self.embedding_model,
                    relation_text,
                    prompt="Instruct: Retrieve descriptions of relations that are present in the given text.\nQuery: ",
                )
            self.target_schema_embedding_dict[relation] = embedding

    def update_schema_embedding_dict(self):
        for relation, relation_definition in self.target_schema_dict.items():
            if relation in self.target_schema_embedding_dict:
                continue
            relation_text = relation_to_text(relation, relation_definition)
            if self.finetuned_e5mistral:
                embedding = llm_utils.get_embedding_e5mistral(
                    self.embedding_model,
                    self.embedding_tokenizer,
                    relation_text,
                )
            else:
                embedding = llm_utils.get_embedding_sts(
                    self.embedding_model,
                    relation_text,
                )
            self.target_schema_embedding_dict[relation] = embedding

        # Keep BM25 index in sync if schema grows
        self._relation_list = list(self.target_schema_dict.keys())
        self._bm25_index = BM25Index.build(self._relation_list) if self.use_bm25 else None

    def retrieve_relevant_relations(
        self,
        query_input_text: str,
        top_k: int = 10,
        bm25_query: Optional[str] = None,
        cross_query: Optional[str] = None,
    ):
        # --- Stage 1: BM25 prefilter by relation name ---
        candidate_relations = list(self.target_schema_embedding_dict.keys())
        if self.use_bm25 and self._bm25_index is not None and self.bm25_top_k and len(candidate_relations) > 0:
            query = bm25_query if bm25_query is not None else query_input_text
            bm25_indices = self._bm25_index.topk(query=query, k=self.bm25_top_k)
            # bm25_indices refer to self._relation_list
            candidate_relations = [self._relation_list[i] for i in bm25_indices]

        target_relation_embedding_list = [self.target_schema_embedding_dict[r] for r in candidate_relations]

        if self.finetuned_e5mistral:
            query_embedding = llm_utils.get_embedding_e5mistral(
                self.embedding_model,
                self.embedding_tokenizer,
                query_input_text,
                "Retrieve descriptions of relations that are present in the given text.",
            )
        else:
            query_embedding = llm_utils.get_embedding_sts(
                self.embedding_model,
                query_input_text,
                prompt="Instruct: Retrieve descriptions of relations that are present in the given text.\nQuery: ",
            )

        # --- Stage 2: bi-encoder similarity ---
        scores = np.array([query_embedding]) @ np.array(target_relation_embedding_list).T

        scores = scores[0]
        highest_score_indices = np.argsort(-scores)

        bi_top_indices = list(highest_score_indices[: min(top_k, len(highest_score_indices))])
        bi_top_relations = [candidate_relations[idx] for idx in bi_top_indices]

        # --- Stage 3: cross-encoder reranking (optional) ---
        if self.cross_encoder is None:
            return bi_top_relations

        ce_k = min(self.cross_top_k, len(bi_top_relations)) if self.cross_top_k else len(bi_top_relations)
        ce_candidates_text = [relation_to_text(r, self.target_schema_dict.get(r)) for r in bi_top_relations[:ce_k]]
        ce_query = cross_query if cross_query is not None else query_input_text
        reranked = self.cross_encoder.rerank(query=ce_query, candidates=ce_candidates_text, top_k=ce_k)

        # reranked indices are relative to the truncated ce_candidates_text
        reranked_relations = [bi_top_relations[i] for (i, _score) in reranked]

        # If cross-encoder returns fewer than top_k, append the remaining bi-encoder results.
        remaining = [r for r in bi_top_relations if r not in set(reranked_relations)]
        return reranked_relations + remaining
