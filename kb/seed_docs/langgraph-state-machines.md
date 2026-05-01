# LangGraph — State Machines for AI Agents

## Definition
LangGraph is a library by LangChain for building stateful, multi-actor applications with language models. It models agent workflows as directed graphs where nodes encode computation (LLM calls, tools, logic) and edges define control flow. A shared state object traverses the graph, enabling cyclic, persistent, and interruptible execution.

## Intuition
Many AI agent tasks require branching, looping, and tool use—patterns that simple chain-of-calls cannot express. LangGraph represents these workflows as a state machine, where each node modifies a persistent state and conditional edges determine the next step. This provides the flexibility of a general program while remaining inspectable and resumable.

## Mathematical Formulation
A LangGraph application is a tuple (G, S, f, c):
- G = (V, E) is a directed graph. V are nodes (agent steps), E are directed edges.
- S is a shared state dictionary.
- Each node v has a function f_v: S → ΔS that returns partial state updates to merge into S.
- Edges may have guard conditions: c_e: S → {True, False} that decide traversal.
- Special edges: START (entry point), END (terminal nodes).
Execution iteratively selects the next node based on current node’s outgoing edges, evaluates conditions, applies node functions, and repeats until an END node is reached. Cyclic edges enable loops (e.g., agent → tool → agent).

## Estimation
Not applicable — LangGraph is a runtime orchestration framework, not a statistical estimator. However, execution behavior (node scheduling, state merging, checkpointing) follows discrete state-machine semantics.

## Assumptions
- Node functions are deterministic enough for predictable execution (idempotency helps recovery).
- State is JSON-serializable to enable checkpointing and persistence.
- LLM calls within nodes are encapsulated and managed externally (e.g., via LangChain model integration).
- Conditional edges rely on interpretable state fields to avoid unintended branches.

## Key Properties
- **Cyclic graphs** support iterative agent patterns (plan, execute, observe, replan).
- **Persistence** via integrated checkpointer saves state after each step, enabling resume and time travel.
- **Human-in-the-loop** by interrupting the graph at any node and awaiting external input.
- **Streaming** of node outputs and partial state for real-time feedback.
- **Composability**: Subgraphs can be embedded as nodes, allowing hierarchical design.

## Failure Modes
- **Infinite loops** if cycle conditions never turn false or termination edges are missing.
- **State explosion** when state accumulates unbounded data across many cycles without pruning.
- **Unintended branching** from ambiguous condition functions, leading to erratic flows.
- **Performance degradation** with many small nodes due to serialization overhead.
- **Non-determinism** from LLM outputs can cause divergent traces that are hard to debug.

## When to Use / Not Use
**Use**: Complex multi-turn agent workflows, tool-using agents, retrieval-augmented loops, conversational agents with state, any process requiring branching and looping with LLM supervision.

**Not use**: Simple linear chains or stateless transformations (overhead is unnecessary), one-off completions, streaming pipelines without decision logic, or when execution must be fully deterministic and low-latency without LLM variability.

## Variants / Extensions
- **LangGraph Studio** provides a visual debugger and editor.
- **Subgraphs** allow packaging a graph as a reusable node.
- **Checkpoint adapters** support different storage backends (SQLite, Postgres, Redis).
- **ToolNode** pre-built node for executing tool calls from an LLM response.
- Integration with **LangSmith** for tracing, evaluation, and monitoring.

## Minimal Example (Python)
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class S(TypedDict):
    count: Annotated[int, operator.add]

def add_one(state: S): return {"count": 1}
def check(state: S): return "done" if state["count"] > 2 else "add"

graph = StateGraph(S)
graph.add_node("add", add_one)
graph.set_entry_point("add")
graph.add_conditional_edges("add", check, {"done": END, "add": "add"})
app = graph.compile()
print(app.invoke({"count": 0}))  # {'count': 3}
```