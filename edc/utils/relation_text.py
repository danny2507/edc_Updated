from __future__ import annotations


def relation_to_text(relation_name: str, relation_definition: str | None) -> str:
    """Build the canonical text representation for a relation.

    We use this for both bi-encoder embeddings and cross-encoder reranking so that
    the scoring stages see consistent candidate text.

    Args:
        relation_name: Canonical relation name (schema key).
        relation_definition: Schema description/definition.

    Returns:
        A single string containing both name and definition.
    """

    relation_name = (relation_name or "").strip()
    relation_definition = (relation_definition or "").strip()

    if relation_definition:
        return f"Relation: {relation_name}. Definition: {relation_definition}"

    # Fallback: at least the model sees the name
    return f"Relation: {relation_name}"
