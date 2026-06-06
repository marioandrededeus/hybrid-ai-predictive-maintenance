from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.llm.semantic_query_router import QUERY_TEMPLATES


EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_SIMILARITY_THRESHOLD = 0.62


@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    """Load the multilingual sentence embedding model once."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_semantic_examples() -> list[dict]:
    """Build a semantic example catalog from approved query templates."""
    examples = []

    for template in QUERY_TEMPLATES:
        intent = template["intent"]

        semantic_examples = template.get("semantic_examples", [])

        if not semantic_examples:
            semantic_examples = template.get("keywords", [])

        for example in semantic_examples:
            examples.append(
                {
                    "intent": intent,
                    "example": example,
                    "sql": template["sql"],
                }
            )

    return examples


@lru_cache(maxsize=1)
def get_embedded_catalog() -> tuple[list[dict], np.ndarray]:
    """Embed all approved semantic examples."""
    model = load_embedding_model()
    examples = get_semantic_examples()

    texts = [item["example"] for item in examples]
    embeddings = model.encode(texts, normalize_embeddings=True)

    return examples, np.array(embeddings)


def route_prompt_by_embedding(
    prompt: str,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> dict:
    """Route a prompt to the closest approved SQL template using embeddings."""
    if not prompt or not prompt.strip():
        return {
            "status": "not_matched",
            "intent": None,
            "sql": None,
            "similarity_score": 0.0,
            "matched_example": None,
            "routing_method": "embedding_similarity",
        }

    model = load_embedding_model()
    examples, catalog_embeddings = get_embedded_catalog()

    prompt_embedding = model.encode([prompt], normalize_embeddings=True)
    similarities = cosine_similarity(prompt_embedding, catalog_embeddings)[0]

    best_index = int(np.argmax(similarities))
    best_score = float(similarities[best_index])
    best_match = examples[best_index]

    if best_score < similarity_threshold:
        return {
            "status": "not_matched",
            "intent": None,
            "sql": None,
            "similarity_score": round(best_score, 4),
            "matched_example": best_match["example"],
            "routing_method": "embedding_similarity",
        }

    return {
        "status": "matched",
        "intent": best_match["intent"],
        "sql": best_match["sql"],
        "similarity_score": round(best_score, 4),
        "matched_example": best_match["example"],
        "routing_method": "embedding_similarity",
    }