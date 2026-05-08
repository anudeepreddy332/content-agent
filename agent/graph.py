"""
agent/graph.py
--------------
LangGraph state machine for the content-agent pipeline.

What this file does:
    Wires all nodes and edges into a compiled LangGraph graph.
    This file contains ONLY graph structure — no logic lives here.
    All node logic lives in nodes.py.

How to run (not directly — use main.py):
    from agent.graph import build_graph
    graph = build_graph()
    result = graph.invoke(initial_state)

Pipeline flow:
    draft → retrieve → verify → reflect → [loop back to draft if needed] → hitl → html_gen → git

Edge logic:
    After reflect_node: route_after_reflect() decides whether to revise or proceed.
    After hitl_node: route_after_hitl() decides approve / reject / feedback loop.

Vulnerabilities:
    - If a node returns a partial state update with wrong keys, LangGraph merges silently.
      This is why AgentState uses TypedDicts with typed fields — catch drift early.
    - Interrupt for HITL (hitl_node) requires graph to be compiled with checkpointer
      in main.py if you want to resume across process restarts. For now, in-process only.
"""

from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import (
    draft_node,
    retrieve_node,
    verify_node,
    reflect_node,
    hitl_node,
    html_gen_node,
    git_node,
    route_after_reflect,
    route_after_hitl,
)

def build_graph() -> StateGraph:
    """
    Build and compile the content-agent LangGraph state machine.

    Returns:
        Compiled LangGraph graph ready for .invoke() or .stream()

    Node registration order doesn't matter - edges define execution order.
    Entry point is always draft_node.
    """

    builder = StateGraph(AgentState)

    # Register all nodes
    builder.add_node("draft", draft_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("verify", verify_node)
    builder.add_node("reflect", reflect_node)
    builder.add_node("hitl", hitl_node)
    builder.add_node("html_gen", html_gen_node)
    builder.add_node("git", git_node)

    # Set entry point
    builder.set_entry_point("draft")

    # Linear edges (no branching)
    builder.add_edge("draft", "retrieve")
    builder.add_edge("retrieve", "verify")
    builder.add_edge("verify", "reflect")

    # Conditional edge after reflect
    # route_after_reflect returns "draft" (revise) or "hitl" (proceed)
    builder.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {
            "draft": "draft",   # revise: loop back
            "hitl": "hitl"  # proceed to human gate
        }
    )

    # Conditional edge after HITL
    # router_after_hitl returns "html_gen" (approved), "draft" (feedback), or END (rejected)
    builder.add_conditional_edges(
        "hitl",
        route_after_hitl,
        {
            "html_gen": "html_gen",
            "draft": "draft",
            END: END,
        }
    )

    # Linear edges to finish
    builder.add_edge("html_gen", "git")
    builder.add_edge("git", END)

    return builder.compile()

















