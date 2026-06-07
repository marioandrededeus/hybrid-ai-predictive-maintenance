import numpy as np

import src.llm.embedding_router as embedding_router


class FakeEmbeddingModel:
    """Small deterministic fake model for unit tests."""

    def encode(self, texts, show_progress_bar=False, normalize_embeddings=True):
        vectors = []

        for text in texts:
            normalized_text = text.lower()

            if any(
                term in normalized_text
                for term in ["machine", "machines", "asset", "assets", "equipment", "critical"]
            ):
                vectors.append([1.0, 0.0, 0.0])

            elif any(
                term in normalized_text
                for term in ["diagnostic", "diagnostics", "severity", "severe"]
            ):
                vectors.append([0.0, 1.0, 0.0])

            else:
                vectors.append([0.0, 0.0, 1.0])

        return np.array(vectors)


def fake_semantic_examples():
    return [
        {
            "intent": "highest_risk_assets",
            "example": "which assets have the highest anomaly risk",
            "approved_question": "Which monitored assets have the highest anomaly risk?",
            "sql": "SELECT 1;",
        },
        {
            "intent": "high_severity_diagnostics",
            "example": "show high severity diagnostics",
            "approved_question": "Show high severity predictive maintenance diagnostics.",
            "sql": "SELECT 2;",
        },
    ]


def setup_fake_embedding_router(monkeypatch):
    embedding_router.get_embedded_catalog.cache_clear()

    monkeypatch.setattr(
        embedding_router,
        "load_embedding_model",
        lambda: FakeEmbeddingModel(),
    )

    monkeypatch.setattr(
        embedding_router,
        "get_semantic_examples",
        fake_semantic_examples,
    )


def test_embedding_router_matches_highest_risk_assets(monkeypatch):
    setup_fake_embedding_router(monkeypatch)

    result = embedding_router.route_prompt_by_embedding(
        "which machines look most critical?",
        similarity_threshold=0.45,
    )

    assert result["status"] == "matched"
    assert result["intent"] == "highest_risk_assets"
    assert result["sql"] is not None
    assert result["routing_method"] == "embedding_similarity"


def test_embedding_router_matches_high_severity_diagnostics(monkeypatch):
    setup_fake_embedding_router(monkeypatch)

    result = embedding_router.route_prompt_by_embedding(
        "show the most severe diagnostic cases",
        similarity_threshold=0.45,
    )

    assert result["status"] == "matched"
    assert result["intent"] == "high_severity_diagnostics"
    assert result["sql"] is not None


def test_embedding_router_blocks_low_similarity_prompt(monkeypatch):
    setup_fake_embedding_router(monkeypatch)

    result = embedding_router.route_prompt_by_embedding(
        "what is the best movie to watch this weekend?",
        similarity_threshold=0.80,
    )

    assert result["status"] == "not_matched"
    assert result["intent"] is None
    assert result["sql"] is None

def test_suggest_approved_questions_returns_ranked_options(monkeypatch):
    setup_fake_embedding_router(monkeypatch)

    suggestions = embedding_router.suggest_approved_questions(
        "which machines look most critical?",
        top_k=2,
    )

    assert isinstance(suggestions, list)
    assert len(suggestions) == 2
    assert suggestions[0]["intent"] == "highest_risk_assets"
    assert suggestions[0]["suggested_question"] == (
        "Which monitored assets have the highest anomaly risk?"
    )
    assert "matched_example" in suggestions[0]
    assert "similarity_score" in suggestions[0]