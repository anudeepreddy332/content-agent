"""
Prompt-level eval for verify_system.md.
Tests: does the prompt reliably return a JSON array with the required fields?
Fixed input — not testing content quality, testing output schema stability.
Run: uv run python evals/prompt_evals/test_verify_prompt.py
"""
import sys
import os
import json
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import DEEPSEEK_MODEL, DEEPSEEK_BASE_URL

VERIFY_SYSTEM = Path("prompts/verify_system.md").read_text(encoding='utf-8')

# Fixed test input — always the same, so results are comparable across prompt versions
FIXED_DRAFT = """
Gradient descent updates parameters by moving opposite to the gradient of the loss.
The learning rate controls step size. A rate of 0.01 is common for neural networks.
Adam optimizer adjusts learning rates per parameter using gradient moments.
"""

FIXED_SOURCES = """
[WEB] https://pytorch.org/docs/stable/optim.html
PyTorch optimizers include SGD, Adam, and RMSprop. Adam uses first and second
moment estimates of gradients to adapt learning rates per parameter.
"""

def run():
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url=DEEPSEEK_BASE_URL)
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": VERIFY_SYSTEM},
            {"role": "user", "content": f"Draft:\n{FIXED_DRAFT}\n\nSources:\n{FIXED_SOURCES}\n\nReturn ONLY the JSON array."}
        ],
        temperature=0.1,
        max_tokens=1000,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith('```'):
        raw = raw.split('```')[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    failures = []
    try:
        parsed = json.loads(raw)
        print("PARSED:", json.dumps(parsed, indent=2)[:2000])

        if not isinstance(parsed, list):
            failures.append(f"Expected list, got {type(parsed)}")
        else:
            for i, item in enumerate(parsed):
                for field in ["claim", "source_url", "confidence", "status"]:
                    if field not in item:
                        failures.append(f"Item {i} missing field: {field}")
                if item.get("status") not in ("verified", "weak", "unverified"):
                    failures.append(f"Item {i} invalid status: {item.get('status')}")
                conf = item.get("confidence", -1)
                if not (0.0 <= conf <= 1.0):
                    failures.append(f"Item {i} confidence out of range: {conf}")

    except json.JSONDecodeError as e:
        failures.append(f"JSON parse failed: {e}")

    cost = (response.usage.prompt_tokens / 1e6 * 0.27) + (response.usage.completion_tokens / 1e6 * 1.10)
    print(f"verify_prompt eval | claims={len(parsed) if not failures else '?'} | cost=${cost:.5f}")
    if failures:
        print("FAIL:")
        for f in failures: print(f"  - {f}")
        sys.exit(1)
    print("PASS")

if __name__ == "__main__":
    run()



