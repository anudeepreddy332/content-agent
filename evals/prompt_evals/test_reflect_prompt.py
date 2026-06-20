"""
Prompt-level eval for reflect_system.md.
Tests: does the prompt reliably return a JSON object with:
       - "score": int, 1–10
       - "notes": non‑empty string

Fixed input — not testing content quality, testing output schema stability.
Run: uv run python evals/prompt_evals/test_reflect_prompt.py
"""
import sys
import os
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from openai import OpenAI
from config import DEEPSEEK_MODEL, DEEPSEEK_BASE_URL

REFLECT_SYSTEM = Path("prompts/reflect_system.md").read_text(encoding='utf-8')

# Fixed test inputs — always the same for reproducibility
FIXED_TOPIC = "Gradient Descent"
FIXED_DRAFT = """
# Gradient Descent
## Problem Framing
You have a model with parameters and a loss function. Gradient descent finds the parameter
values that minimize that loss by iteratively moving downhill in the loss landscape. It is
the core optimization algorithm behind most modern machine learning. Without it, training
neural networks with millions of parameters would be computationally infeasible.

## Technical Deep-Dive
Gradient descent works by computing the gradient of the loss with respect to each parameter,
then updating parameters in the opposite direction. The learning rate controls step size.
Common variants include batch, stochastic, and mini-batch gradient descent. Adam is the
default optimizer for most deep learning tasks because it adapts learning rates per parameter.

## Code
```python
w = w - lr * dw
b = b - lr * db
```
## Takeaways
Gradient descent finds the minimum of a loss function by following the negative gradient

The learning rate is the most important hyperparameter to tune

Adam is the default optimizer for deep learning
"""
FIXED_GROUNDING_SUMMARY = "3 verified, 2 weak, 5 unverified out of 10 total claims."

def run():
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=DEEPSEEK_BASE_URL,
    )
    user_message = f"""
Topic: {FIXED_TOPIC}

Series context: Learning log

Draft: {FIXED_DRAFT}

Grounding report summary: {FIXED_GROUNDING_SUMMARY}
Grounding score: 0.60

Evaluate this draft on:
1. Structure — are all required sections present and coherent
2. Technical depth — senior-engineer level, not entry-level blog post
3. Source grounding — claims are backed by evidence
4. Writing quality — direct, no hedging, no AI filler phrases

Return ONLY JSON: {{"score": <int 1-10>, "notes": "<2-3 sentences of specific critique>"}}

"""
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": REFLECT_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=500,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith('```'):
        raw = raw.split('```')[1]
        if raw.startswith('json'):
            raw = raw[4:]
        raw = raw.strip()
    failures = []
    parsed = {}

    try:
        parsed = json.loads(raw)
        print("PARSED:", json.dumps(parsed, indent=2)[:2000])

    except json.JSONDecodeError as e:
        failures.append(f"JSON parse failed: {e}")
        cost = (response.usage.prompt_tokens / 1e6 * 0.27) + (
            response.usage.completion_tokens / 1e6 * 1.10
        )
        print(f"reflect_prompt eval | cost=${cost:.5f}")
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    score = parsed.get("score")
    if score is None:
        failures.append("Missing key: Score")
    elif not isinstance(score, int):
        failures.append(f"score must be int, got {type(score)}: {score}")
    elif not (1 <= score <= 10):
        failures.append(f"score out of range (1-10): {score}")

    notes = parsed.get("notes")
    if notes is None:
        failures.append("Missing key: notes")
    elif not isinstance(notes, str):
        failures.append(f"notes must be string, got {type(notes)}")
    elif not notes.strip():
        failures.append("notes is empty or whitespace only")

    cost = (response.usage.prompt_tokens / 1e6 * 0.27) + (
        response.usage.completion_tokens / 1e6 * 1.10
    )

    print(f"reflect_prompt eval | score={score} | cost=${cost:.5f}")

    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print("PASS")
    sys.exit(0)

if __name__ == "__main__":
    run()





