# ReAct Agent Pattern

## Definition
The ReAct (Reasoning and Acting) pattern is an agent design where a language model generates interleaved reasoning traces and executable actions. The model produces a thought explaining what it intends to do, then an action command; the environment executes the action and returns an observation, which is appended to the context, and the loop continues until a final answer is output.

## Intuition
Pure chain-of-thought reasoning operates only on the model’s internal knowledge, leading to hallucinations. Pure acting (tool use) lacks explicit planning. ReAct combines them: the model thinks about what it needs, performs actions to gather external information, then reasons over the observations, dynamically adapting its plan.

## Mathematical Formulation
At step t, the agent maintains a history H_t = [x, (T_0, A_0, O_0), …, (T_t, A_t, O_t)] where x is the user query, T_i is a reasoning thought, A_i is a parsable action, and O_i is the observation returned by the environment. The LLM acts as a stochastic policy π sampling (T_t, A_t) ~ π(·| H_{t-1}). The environment is a deterministic transition function E: A_t → O_t. The loop terminates when A_t is a special “finish” action with the final answer, or after a maximum number of steps. The final answer is extracted from the last T_t.

## Estimation
ReAct is typically a prompting method, not a trained estimator. The LLM is prompted with few-shot examples demonstrating the thought-action-observation interleaving. No parameter update is required. Fine-tuning variants train the model on ReAct-augmented trajectories to improve action validity and reasoning coherence, using standard language-modeling objectives on token sequences that include special delimiters for thoughts and actions.

## Assumptions
- The model can reliably generate structured actions parsable by the environment (e.g., JSON, specific command text).
- The environment provides informative textual observations for the model to interpret.
- The action space is well-defined and the prompt conveys the available tools.
- Few-shot examples correctly illustrate the desired interleaving format.

## Key Properties
- **Dynamic information seeking**: The model can decide when and what external information to fetch.
- **Improved factual grounding**: Explicitly retrieved observations replace hallucinated knowledge.
- **Interpretability**: The reasoning trace exposes the decision process.
- **Synergy**: Reasoning helps guide actions; action observations correct and inform the next reasoning step.
- **LLM agnostic**: Works with any instruction-following model capable of structured generation.

## Failure Modes
- **Action parsing errors**: Model produces malformed commands, breaking the execution loop.
- **Reasoning-action misalignment**: Thought describes one intent but action does something else.
- **Infinite loops**: Model never emits the finish action, cycling repetitively on the same action.
- **Fabricated observations**: Model may hallucinate the outcome of an action instead of waiting for the real observation.
- **Latency**: Multiple serial LLM calls increase response time.

## When to Use / Not Use
**Use**: Multi-hop question answering, interactive web navigation, API orchestration, complex research tasks requiring iterative lookups, debugging with external tools.

**Not use**: Simple factual queries answerable in one step, latency-critical applications, environments where actions have irreversible side effects without human confirmation, or when the model’s instruction-following ability is insufficient to maintain the pattern.

## Variants / Extensions
- **ReAct + Reflexion**: Adds a posterior verbal critique step to improve future attempts.
- **Tree-of-Thought + ReAct**: Expands the reasoning into a search tree, interleaving actions at nodes.
- **ReWOO**: Decouples reasoning and actions into a planning phase, then parallel execution, reducing LLM calls.
- **Fine-tuned ReAct**: Models like ReAct-T5 are trained explicitly to produce thought-action-observation sequences.
- **Tool-use augmentation**: Integrating ReAct pattern into chat-optimized models via function calling (OpenAI functions).

## Minimal Example (Python)
```python
import openai

def react_agent(query, tools, max_steps=5):
    history = [f"Question: {query}"]
    for _ in range(max_steps):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "\n".join(history)}],
            stop=["Observation:"]
        )["choices"][0]["message"]["content"]
        thought, action = response.split("\nAction: ")
        history.append(thought)
        history.append(f"Action: {action}")
        # execute action via tools dict (not shown)
        obs = tools.get(action.strip(), "No such tool")
        history.append(f"Observation: {obs}")
        if "Finish" in action:
            return obs
    return "Max steps reached"
```