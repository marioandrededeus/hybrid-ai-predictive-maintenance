"""
Domain guard for the Hybrid AI Predictive Maintenance project.

This module validates whether a user question belongs to the expected
industrial predictive maintenance context before allowing it to reach
Text-to-SQL, agents, or future LLM calls.
"""


ALLOWED_DOMAIN_KEYWORDS = [
    "maintenance",
    "predictive maintenance",
    "vibration",
    "asset",
    "machine",
    "equipment",
    "sensor",
    "measurement",
    "scenario",
    "diagnostic",
    "diagnosis",
    "anomaly",
    "anomaly score",
    "anomaly probability",
    "risk",
    "risk level",
    "severity",
    "recommendation",
    "human validation",
    "validation",
    "rms",
    "peak velocity",
    "crest factor",
    "temperature",
    "load",
    "frequency",
    "spectral",
    "spectrum",
    "broadband",
    "low frequency",
    "high frequency",
    "harmonic",
    "subharmonic",
    "carpet",
    "lubrication",
    "structural looseness",
    "looseness",
    "normal operation",
    "ml diagnostic",
    "model",
    "prediction",
]

BLOCKED_GENERAL_KEYWORDS = [
    "capital",
    "weather",
    "football",
    "soccer",
    "world cup",
    "recipe",
    "movie",
    "song",
    "joke",
    "marketing",
    "instagram",
    "politics",
    "president",
    "stock price",
    "crypto",
    "travel",
]


def normalize_prompt(prompt: str) -> str:
    """
    Normalize user prompt for simple rule-based validation.
    """

    return prompt.strip().lower()


def is_prompt_in_domain(prompt: str) -> bool:
    """
    Check whether the prompt belongs to the predictive maintenance domain.
    """

    normalized_prompt = normalize_prompt(prompt)

    if not normalized_prompt:
        return False

    has_allowed_keyword = any(
        keyword in normalized_prompt
        for keyword in ALLOWED_DOMAIN_KEYWORDS
    )

    has_blocked_keyword = any(
        keyword in normalized_prompt
        for keyword in BLOCKED_GENERAL_KEYWORDS
    )

    if has_blocked_keyword and not has_allowed_keyword:
        return False

    return has_allowed_keyword


def get_domain_guard_response(prompt: str) -> dict:
    """
    Return a structured validation response for the user prompt.
    """

    is_allowed = is_prompt_in_domain(prompt)

    if is_allowed:
        return {
            "is_allowed": True,
            "reason": "Prompt is related to the predictive maintenance domain.",
            "message": "Prompt accepted."
        }

    return {
        "is_allowed": False,
        "reason": "Prompt is outside the predictive maintenance domain.",
        "message": (
            "This assistant only answers questions related to predictive "
            "maintenance, vibration diagnostics, industrial assets, anomaly "
            "detection, risk analysis, and maintenance recommendations."
        )
    }