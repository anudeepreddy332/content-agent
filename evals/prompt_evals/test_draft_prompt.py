"""
Prompt-level eval for draft_system.md.
test_draft_prompt.py      ← Does draft_system.md produce valid JSON with all 4 keys?
Fixed test inputs — not testing content quality, testing output schema stability.
Run: uv run python evals/prompt_evals/test_draft_prompt.py
"""

import sys, json, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from openai import OpenAI
from config import DEEPSEEK_MODEL, DEEPSEEK_BASE_URL, DRAFT_TEMPERATURE

DRAFT_SYSTEM = Path("prompts/draft_system.md").read_text(encoding='utf-8')

# Fixed inputs
FIXED_TOPIC = "Gradient Descent"
FIXED_SERIES = "Concept Exploration"
FIXED_CARD_ID = "TEST-01"

def run():
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url=DEEPSEEK_BASE_URL)

    user_message = f"""Write a technical article for The Machinist on the following topic.

    Topic: {FIXED_TOPIC}
    Card ID: {FIXED_CARD_ID}
    Series context: {FIXED_SERIES}
    Audience: engineers learning ML and agentic AI.
    Return ONLY the JSON object as specified in your instructions. No markdown wrapper."""

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": DRAFT_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        temperature=DRAFT_TEMPERATURE,
        max_tokens=4000,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if the model wrapped the JSON anyway
    if raw.startswith('```'):
        raw = raw.split('```')[1]
        if raw.startswith('json'):
            raw = raw[4:]
        raw = raw.strip()

    failures = []
    parsed = {}

    try:
        parsed = json.loads(raw)
        print("PARSED:", json.dumps(parsed, indent=2)[:4000])

    except json.JSONDecodeError as e:
        failures.append(f"JSON parse failed: {e}")
        cost = (response.usage.prompt_tokens / 1e6 * 0.27) + (response.usage.completion_tokens / 1e6 * 1.10)
        print(f"draft_prompt eval | cost=${cost:.5f}")
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    # Check that all four required keys exist and are non‑empty strings
    required_keys = ["problem_framing", "technical_dive", "code_snippets", "takeaways"]
    for key in required_keys:
        if key not in parsed:
            failures.append(f"Missing key: {key}")
        else:
            value = parsed[key]
            if not isinstance(value, str) or not value.strip():
                failures.append(f"Key '{key}' is empty or not a string")

    # Compute cost for observability
    cost = (response.usage.prompt_tokens / 1e6 * 0.27) + (
            response.usage.completion_tokens / 1e6 * 1.10
    )
    print(f"draft_prompt eval | keys={len(parsed)} | cost=${cost:.5f}")

    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print("PASS")
    sys.exit(0)


if __name__ == "__main__":
    run()




