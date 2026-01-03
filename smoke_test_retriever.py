from sentence_transformers import SentenceTransformer

from edc.schema_retriever import SchemaRetriever


def main() -> None:
    # tiny model for quick smoke test
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    schema = {
        "country": "The subject entity is located in the country specified by the object entity.",
        "date of birth": "The subject entity was born on the date specified by the object entity.",
        "place of birth": "The subject entity was born in the location specified by the object entity.",
    }

    sr = SchemaRetriever(
        schema,
        model,
        None,
        finetuned_e5mistral=False,
        bm25_top_k=2,
        use_bm25=True,
        cross_encoder_model_name=None,
    )

    res = sr.retrieve_relevant_relations("Barack Obama was born in Hawaii.", top_k=2)
    print(res)


if __name__ == "__main__":
    main()
