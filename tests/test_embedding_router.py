from src.llm.embedding_router import route_prompt_by_embedding


def test_embedding_router_matches_highest_risk_assets():
    result = route_prompt_by_embedding(
        "which machines look most critical?",
        similarity_threshold=0.45,
    )

    assert result["status"] == "matched"
    assert result["intent"] == "highest_risk_assets"
    assert result["sql"] is not None
    assert result["routing_method"] == "embedding_similarity"


def test_embedding_router_matches_high_severity_diagnostics():
    result = route_prompt_by_embedding(
        "show the most critical diagnostic cases",
        similarity_threshold=0.45,
    )

    assert result["status"] == "matched"
    assert result["intent"] == "high_severity_diagnostics"
    assert result["sql"] is not None


def test_embedding_router_blocks_low_similarity_prompt():
    result = route_prompt_by_embedding(
        "what is the best movie to watch this weekend?",
        similarity_threshold=0.80,
    )

    assert result["status"] == "not_matched"
    assert result["intent"] is None
    assert result["sql"] is None