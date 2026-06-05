"""
Domain guard for the Hybrid AI Predictive Maintenance project.

This module validates whether a user question belongs to the expected
industrial predictive maintenance context before allowing it to reach
Text-to-SQL, agents, or future LLM calls.
"""


CORE_MAINTENANCE_KEYWORDS = [
    "maintenance",
    "predictive maintenance",
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
    "ml diagnostic",
    "prediction",
]

INDUSTRIAL_CONTEXT_KEYWORDS = [
    "asset",
    "assets",
    "machine",
    "machines",
    "equipment",
    "sensor",
    "sensors",
    "measurement",
    "measurements",
    "scenario",
    "scenarios",
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
    "vibration",
]

BLOCKED_OUT_OF_SCOPE_KEYWORDS = [
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
    "vulcan",
    "volcano",
    "etna",
    "earthquake",
    "seismic",
    "geology",
    "mountain",
]


def normalize_prompt(prompt: str) -> str:
    """
    Normalize user prompt for simple rule-based validation.
    """

    return prompt.strip().lower()


def is_prompt_in_domain(prompt: str) -> bool:
    """
    Check whether the prompt belongs to the predictive maintenance domain.

    A prompt is allowed only when it is clearly related to the project context.
    Generic words like "vibration" are not enough by themselves.
    """

    normalized_prompt = normalize_prompt(prompt)

    if not normalized_prompt:
        return False

    has_blocked_keyword = any(
        keyword in normalized_prompt
        for keyword in BLOCKED_OUT_OF_SCOPE_KEYWORDS
    )

    if has_blocked_keyword:
        return False

    has_core_keyword = any(
        keyword in normalized_prompt
        for keyword in CORE_MAINTENANCE_KEYWORDS
    )

    has_industrial_context = any(
        keyword in normalized_prompt
        for keyword in INDUSTRIAL_CONTEXT_KEYWORDS
    )

    return has_core_keyword and has_industrial_context


def get_domain_guard_response(prompt: str) -> dict:
    """
    Return a structured validation response for the user prompt.
    """

    is_allowed = is_prompt_in_domain(prompt)

    if is_allowed:
        return {
            "is_allowed": True,
            "reason": "Prompt is related to the predictive maintenance domain.",
            "message": "Prompt accepted.",
        }

    return {
        "is_allowed": False,
        "reason": "Prompt is outside the predictive maintenance domain.",
        "message": (
            "This assistant only answers questions related to industrial "
            "predictive maintenance, vibration diagnostics, monitored assets, "
            "anomaly detection, risk analysis, and maintenance recommendations."
        ),
    }