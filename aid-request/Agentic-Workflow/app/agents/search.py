import json
import re
from typing import TypedDict, List, Dict, Any, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AnyMessage
# Crucial Import: Tells LangGraph to APPEND messages instead of OVERWRITING them
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.state import CaseState
from app.services.llm import llm_model 
from app.tools.search_tools import medical_search, tavily_search

# Local sub-graph state schema
class SearchAgentState(TypedDict):
    text: str
    # FIX: This Annotation forces LangGraph to maintain a historical log of tool results
    messages: Annotated[List[AnyMessage], add_messages]
    search_results: Dict[str, Any]
    loop_count: int

# Define and bind tools
tools = [medical_search, tavily_search]
model_with_tools = llm_model.bind_tools(tools)

def call_search_model(state: SearchAgentState):
    system_prompt = """أنت وكيل ذكي مخصص للبحث والتحليل بناءً على 3 حالات محددة فقط. قم بتحليل النص واستخدم الأدوات المتاحة لتلبية الطلب بدقة:

1. مصطلحات معرفية متخصصة (Domain-Specific Knowledge):
   - إذا تم ذكر مصطلح تقني أو علمي أو هندسي أو تخصصي، استخدم أداة البحث (tavily_search) لجلب تعريفه وشرحه الدقيق.

2. تسعير السلع والخدمات والمنتجات (Pricing):
   - إذا طُلب تسعير أي مادة، أداة، عقار، خدمة، أو عنصر يمكن شراؤه أو استئجاره، قم بالبحث عن أسعاره الحالية في السوق.

3. الاستعلامات الطبية (Medical Queries):
   - إذا تم ذكر اسم دواء: استخدم أداة (medical_search) أولاً في قاعدة البيانات المحلية لمعرفة دواعي الاستعمال (usage causes) والأسعار المتاحة.
   - إذا تم ذكر حالة مرضية أو أعراض (Medical Case): استخدم أولاً أداة (medical_search) أو (tavily_search) للبحث عن الأدوية المقابلة والمناسبة لهذه الحالة، ثم ابحث لاحقاً عن أسعار تلك الأدوية في السوق.

⚠️ قواعد العمل والتنفيذ:
- للبحث الطبي، التزم بالترتيب: ابحث عن الدواء/الحالة في قاعدة البيانات المحلية أولاً، ثم انتقل للتسعير عبر الإنترنت إن لم تجد السعر محلياً.
- إذا أعطتك الأدوات إجابات واضحة، لا تكرر البحث أبداً! صغ المخرج النهائي فوراً.
- بعد الانتهاء تماماً، صغ المخرج النهائي في قالب JSON صالح فقط دون أي مقدمات أو نصوص جانبية كالتالي:
{
  "definitions": [{"term": "المصطلح", "definition": "التعريف المشروح"}],
  "medical_analysis": [{"case_or_drug": "الحالة أو الدواء", "usage_causes": "دواعي الاستعمال أو الأدوية المقابلة", "source": "المصدر"}],
  "pricing": [{"item": "الشيء المسعر", "price": "السعر أو نطاق السعر المكتشف", "type": "purchase or rent", "sources": ["روابط المصادر"]}]
}"""
    
    # Prepend system prompt cleanly to the current historical messages array
    full_messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model_with_tools.invoke(full_messages)
    
    current_loops = state.get("loop_count", 0) + 1
    
    # Returning the single message updates the annotated list by appending it
    return {
        "messages": [response],
        "loop_count": current_loops
    }


def should_continue(state: SearchAgentState):
    if state.get("loop_count", 0) >= 6:
        return "parse_output"
        
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "parse_output"

def clean_and_parse_json(text_content: str) -> Dict[str, Any]:
    """
    Robust JSON extractor and sanitizer. Removes markdown, conversational text,
    trailing commas, and attempts to parse safely.
    """
    if not text_content:
        return {}

    # Step 1: Remove markdown code block wrappers (e.g., ```json or ```)
    clean_text = re.sub(r"```(?:json)?\s*", "", text_content)
    clean_text = re.sub(r"```", "", clean_text).strip()

    # Step 2: Extract content inside the outermost curly braces
    json_match = re.search(r"(\{[\s\S]*\})", clean_text)
    if not json_match:
        return {}

    extracted_json_str = json_match.group(1).strip()

    # Step 3: Clean trailing commas before closing brackets or braces (common LLM generation mistake)
    # e.g., [1, 2, ] -> [1, 2] or {"a": 1, } -> {"a": 1}
    extracted_json_str = re.sub(r",\s*([\]}])", r"\1", extracted_json_str)

    try:
        return json.loads(extracted_json_str)
    except json.JSONDecodeError:
        # Step 4: Final desperate cleanup attempt: Strip any leading/trailing invalid control chars
        try:
            # Re-attempt with raw character replacement
            sanitized = extracted_json_str.replace('\n', ' ').replace('\r', '')
            return json.loads(sanitized)
        except Exception:
            return {}


def parse_search_output(state: SearchAgentState):
    raw_content = state["messages"][-1].content
    parsed_json = clean_and_parse_json(raw_content)
    
    # Validate structure. If keys are missing, populate them correctly
    if not parsed_json or not any(k in parsed_json for k in ["definitions", "medical_analysis", "pricing"]):
        parsed_json = {
            "definitions": parsed_json.get("definitions", []),
            "medical_analysis": parsed_json.get("medical_analysis", []),
            "pricing": parsed_json.get("pricing", []),
            "raw_fallback": raw_content
        }
    else:
        # Ensure fallback key isn't left hanging in valid parsed JSON
        parsed_json.pop("raw_fallback", None)
        
    return {"search_results": parsed_json}

# --- BUILD INTERNAL SUB-GRAPH ---
sub_builder = StateGraph(SearchAgentState)
sub_builder.add_node("call_model", call_search_model)
sub_builder.add_node("tools", ToolNode(tools))
sub_builder.add_node("parse_output", parse_search_output)

sub_builder.set_entry_point("call_model")
sub_builder.add_conditional_edges(
    "call_model",
    should_continue,
    {
        "tools": "tools",
        "parse_output": "parse_output"
    }
)
sub_builder.add_edge("tools", "call_model")
sub_builder.add_edge("parse_output", END)

compiled_search_agent = sub_builder.compile()


# --- THE INTERFACE NODE FOR THE MAIN GRAPH ---
def search_agent(state: CaseState) -> dict:
    """The wrapper function imported by the main orchestrator graph."""
    # query = state.get("reasoning", {}).get("question_or_query", state.get("text", ""))
    reasoning = state.get("reasoning", {})
    query = reasoning.get("question_or_query", state.get("text", ""))
    why = reasoning.get("reasoning", "")

    augmented_query = f"""
    USER INTENT:
    {query}

    REASON FOR SEARCH:
    {why}
    """
    sub_graph_input = {
        "text": augmented_query,
        "messages": [HumanMessage(content=augmented_query)],
        "search_results": {},
        "loop_count": 0
    }
    
    output = compiled_search_agent.invoke(sub_graph_input)
    results = output["search_results"]
    
    current_evidence = state.get("evidence") if isinstance(state.get("evidence"), dict) else {}
    current_evidence["search"] = results
    
    history = state.get("inquiry_history", [])

    history.append({
        "type": "search",
        "target": augmented_query,
        "content": json.dumps(results, ensure_ascii=False)
    })

    return {
        "search_results": json.dumps(results, ensure_ascii=False),
        "evidence": current_evidence,
        "inquiry_history": history
    }