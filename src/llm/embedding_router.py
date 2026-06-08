from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.llm.semantic_query_router import QUERY_TEMPLATES


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_SIMILARITY_THRESHOLD = 0.62
TECHNICAL_TERM_BOOST = 0.20


TECHNICAL_TERM_INTENT_HINTS = {
    "structural_looseness_cases": [
        "structural looseness",
        "looseness",
        "mechanical looseness",
        "loose structure",
        "loose foundation",
        "foundation looseness",
        "loose bolts",
        "bolt looseness",
        "low frequency energy",
        "harmonic ratio",
        "subharmonic ratio",
        "folga estrutural",
        "folga mecanica",
        "folga mecânica",
        "estrutura solta",
        "base solta",
        "fundacao solta",
        "fundação solta",
        "parafuso solto",
        "parafusos soltos",
        "energia de baixa frequencia",
        "energia de baixa frequência",
        "razao harmonica",
        "razão harmônica",
        "relacao harmonica",
        "relação harmônica",
        "sub-harmonica",
        "sub-harmônica",
        "holgura estructural",
        "holgura mecanica",
        "holgura mecánica",
        "estructura suelta",
        "base suelta",
        "pernos sueltos",
        "energia de baja frecuencia",
        "energía de baja frecuencia",
        "relacion armonica",
        "relación armónica",
        "subarmonica",
        "subarmónica",
    ],
    "lubrication_issues": [
        "lubrication",
        "lubrication issue",
        "lubrication problem",
        "lubrication failure",
        "starved lubrication",
        "poor lubrication",
        "lack of lubrication",
        "carpet",
        "carpet pattern",
        "broadband energy",
        "high frequency energy",
        "high-frequency energy",
        "lubrificacao",
        "lubrificação",
        "problema de lubrificacao",
        "problema de lubrificação",
        "falha de lubrificacao",
        "falha de lubrificação",
        "falta de lubrificacao",
        "falta de lubrificação",
        "lubrificacao insuficiente",
        "lubrificação insuficiente",
        "padrao carpet",
        "padrão carpet",
        "energia broadband",
        "energia de alta frequencia",
        "energia de alta frequência",
        "lubricacion",
        "lubricación",
        "problema de lubricacion",
        "problema de lubricación",
        "falla de lubricacion",
        "falla de lubricación",
        "falta de lubricacion",
        "falta de lubricación",
        "lubricacion insuficiente",
        "lubricación insuficiente",
        "patron carpet",
        "patrón carpet",
        "energia de alta frecuencia",
        "energía de alta frecuencia",
    ],
}


@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    """Load the sentence embedding model once."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_semantic_examples() -> list[dict]:
    """
    Build a semantic example catalog from approved query templates.

    The catalog uses semantic_examples when available.
    If semantic_examples are not defined for a template, it falls back to keywords.
    """

    examples = []

    for template in QUERY_TEMPLATES:
        intent = template["intent"]
        approved_question = template.get("approved_question")

        semantic_examples = template.get("semantic_examples", [])

        if not semantic_examples:
            semantic_examples = template.get("keywords", [])

        for example in semantic_examples:
            examples.append(
                {
                    "intent": intent,
                    "example": example,
                    "approved_question": approved_question or example,
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

    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    return examples, np.array(embeddings)


def apply_technical_term_boost(prompt: str, examples: list[dict], similarities: np.ndarray) -> np.ndarray:
    """
    Boost similarity scores when explicit technical fault terms appear in the prompt.

    This does not execute SQL and does not bypass the Domain Guard.
    It only improves ranking among approved intents.
    """

    adjusted_similarities = similarities.copy()
    normalized_prompt = prompt.lower()

    for index, example in enumerate(examples):
        intent = example["intent"]
        technical_terms = TECHNICAL_TERM_INTENT_HINTS.get(intent, [])

        if any(term in normalized_prompt for term in technical_terms):
            adjusted_similarities[index] = min(
                float(adjusted_similarities[index]) + TECHNICAL_TERM_BOOST,
                1.0,
            )

    return adjusted_similarities


def route_prompt_by_embedding(
    prompt: str,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> dict:
    """
    Route a prompt to the closest approved SQL template using embeddings.

    The embedding model only selects an approved intent.
    It does not generate SQL.
    """

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

    prompt_embedding = model.encode(
        [prompt],
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    similarities = cosine_similarity(prompt_embedding, catalog_embeddings)[0]
    similarities = apply_technical_term_boost(prompt, examples, similarities)

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


def suggest_approved_questions(prompt: str, top_k: int = 3) -> list[dict]:
    """
    Suggest approved questions for an ambiguous prompt.

    This function does not execute SQL.
    It only returns the closest approved questions based on embedding similarity.
    """

    if not prompt or not prompt.strip():
        return []

    model = load_embedding_model()
    examples, catalog_embeddings = get_embedded_catalog()

    prompt_embedding = model.encode(
        [prompt],
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    similarities = cosine_similarity(prompt_embedding, catalog_embeddings)[0]
    similarities = apply_technical_term_boost(prompt, examples, similarities)

    ranked_indices = np.argsort(similarities)[::-1]

    suggestions = []
    seen_questions = set()

    for index in ranked_indices:
        example = examples[int(index)]
        approved_question = example["approved_question"]

        if approved_question in seen_questions:
            continue

        suggestions.append(
            {
                "intent": example["intent"],
                "suggested_question": approved_question,
                "matched_example": example["example"],
                "similarity_score": round(float(similarities[index]), 4),
            }
        )

        seen_questions.add(approved_question)

        if len(suggestions) >= top_k:
            break

    return suggestions