from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

print("Starting manual embedding router test...", flush=True)
print(f"Project root: {PROJECT_ROOT}", flush=True)

from src.llm.embedding_router import EMBEDDING_MODEL_NAME, route_prompt_by_embedding

print("Embedding router imported successfully.", flush=True)
print(f"Embedding model: {EMBEDDING_MODEL_NAME}", flush=True)


TEST_PROMPTS = [
    "which machines look most critical?",
    "show equipment with abnormal behavior",
    "which diagnostics require urgent attention?",
    "compare vibration intensity across scenarios",
    "qual cenário tem maior nível médio de vibração?",
    "mostrar equipamentos com comportamento anormal",
    "what is the best movie to watch this weekend?",
]


def main() -> None:
    print("Running test prompts...", flush=True)

    for prompt in TEST_PROMPTS:
        print("\n" + "=" * 80, flush=True)
        print("Prompt:", prompt, flush=True)
        print("Calling embedding router...", flush=True)

        result = route_prompt_by_embedding(prompt)

        print("Status:", result["status"], flush=True)
        print("Intent:", result["intent"], flush=True)
        print("Score:", result["similarity_score"], flush=True)
        print("Matched example:", result["matched_example"], flush=True)


if __name__ == "__main__":
    main()