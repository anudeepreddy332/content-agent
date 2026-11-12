"""Production graph: frozen quality episodes, two human gates, guarded Git."""
from functools import partial

from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent import workflow
from agent.nodes import (
    draft_node, retrieve_node, verify_node, reflect_node, hitl_node,
    html_gen_node, git_node, hitl_html_node, html_revise_node,
    route_after_draft, route_after_verify, route_after_reflect,
    route_after_hitl, route_after_html_review,
)


def _continue(state):
    return END if state.get("terminal_status") else "continue"


def build_graph(checkpointer=None):
    builder = StateGraph(AgentState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("start_quality_episode", workflow.start_episode)
    builder.add_node("draft", partial(workflow.draft_step, run=draft_node))
    builder.add_node("verify", partial(workflow.verify_step, run=verify_node))
    builder.add_node("reflect", partial(workflow.reflect_step, run=reflect_node))
    builder.add_node("hitl", hitl_node)
    builder.add_node("html_gen", partial(workflow.html_step, run=html_gen_node))
    builder.add_node("hitl_html", hitl_html_node)
    builder.add_node("html_revise", partial(workflow.layout_step, run=html_revise_node))
    builder.add_node("git", git_node)
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "start_quality_episode")
    builder.add_conditional_edges("start_quality_episode", _continue, {"continue": "draft", END: END})
    builder.add_conditional_edges("draft", route_after_draft, {
        "verify": "verify", "draft": "draft", END: END,
    })
    builder.add_conditional_edges("verify", route_after_verify, {
        "reflect": "reflect", "draft": "draft", END: END,
    })
    builder.add_conditional_edges("reflect", route_after_reflect, {
        "draft": "draft", "hitl": "hitl", END: END,
    })
    builder.add_conditional_edges("hitl", route_after_hitl, {
        "html_gen": "html_gen", "draft": "draft", END: END,
    })
    builder.add_conditional_edges("html_gen", _continue, {"continue": "hitl_html", END: END})
    builder.add_conditional_edges("hitl_html", route_after_html_review, {
        "git": "git", "html_revise": "html_revise", END: END,
    })
    builder.add_conditional_edges("html_revise", _continue, {"continue": "hitl_html", END: END})
    builder.add_edge("git", END)
    return builder.compile(checkpointer=checkpointer)
