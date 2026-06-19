from langgraph.graph import StateGraph, END
from app.state import CaseState
from app.nodes.intake import intake_node
from app.nodes.evidence import evidence_node
from app.nodes.vqa import vqa_node
from app.nodes.reasoning import reasoning_node
from app.nodes.report import report_node
from app.agents.search import search_agent
def decide_next_step(state: CaseState):
    loop_count = state.get("loop_count", 0)

    reasoning = state.get("reasoning", {})
    next_step = reasoning.get("next_step")

    history = state.get("inquiry_history", [])
    last_step_type = None

    if history:
        last_item = history[-1]

        if isinstance(last_item, dict):
            last_step_type = last_item.get("type")

    # stop after 3 tool calls
    if loop_count >= 3:
        return "report"

    # LLM explicitly requested report
    if next_step == "report":
        return "report"

    # prevent same tool repeatedly
    if (
        last_step_type
        and next_step == last_step_type
        and next_step in {"vqa", "search"}
    ):
        return "report"

    return next_step if next_step in {
        "vqa",
        "search",
        "report"
    } else "report"

builder = StateGraph(CaseState)
builder.add_node("intake", intake_node)
builder.add_node("evidence", evidence_node)
builder.add_node("vqa", vqa_node)
builder.add_node("search", search_agent)
builder.add_node("reasoning", reasoning_node)
builder.add_node("report", report_node)

builder.set_entry_point("intake")
builder.add_edge("intake", "evidence")
builder.add_edge("evidence", "reasoning")

builder.add_conditional_edges(
    "reasoning",
    decide_next_step,
    {"vqa": "vqa", "search": "search", "report": "report"}
)

builder.add_edge("vqa", "reasoning")
builder.add_edge("search", "reasoning")
builder.add_edge("report", END)

graph = builder.compile()