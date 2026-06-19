import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import the actual agents you wrote
# from app.nodes.intake import intake_node
from app.agents.search import search_agent

# 1. Define the minimal CaseState schema for the test runner
class TestCaseState(TypedDict):
    text: Optional[str]
    voice_path: Optional[str]
    images: List[str]
    transcript: Optional[str]
    normalized_case: Dict[str, Any]
    search_results: Optional[str]
    evidence: Dict[str, Any]

# 2. Build a Short-Circuited Test Graph (Intake -> Search -> END)
test_builder = StateGraph(TestCaseState)

# test_builder.add_node("intake", intake_agent)
test_builder.add_node("search", search_agent)

test_builder.set_entry_point("search")
# test_builder.add_edge("intake", "search")
test_builder.add_edge("search", END)  # Bypass reasoning and reviewer for now

test_graph = test_builder.compile()

# =========================================================
# RUN THE TEST
# =========================================================
if __name__ == "__main__":
    print("🚀 Initializing Multi-Modal Mock Pipeline...")

    # Mock sample environment files if you want to test the tool branches
    # In production, replace these paths with your real test assets
    sample_voice = "data/v333.mp3"
    sample_image = "data/prescription.jpg"
    
    # Let's create empty dummy files if they don't exist just so the os.path.exists checks pass
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(sample_voice):
        with open(sample_voice, "w") as f: f.write("dummy audio")
    if not os.path.exists(sample_image):
        with open(sample_image, "w") as f: f.write("dummy image data")

    # Define your composite mock state input
    # Define your composite mock state input with explicit targets
    initial_state = {
        "text": "والدي يعاني من أعراض جلطة وبياخد أسبيرين. ومحتاجين نعرف سعر كرسي متحرك كهربائي للشراء في مصر  ",
        "voice_path": None,
        "images": [],
        "normalized_case": {},
        "evidence": {}
    }

    print("\n--- Running Direct Search Agent Test ---")
    final_state = test_graph.invoke(initial_state)
    print("--- Pipeline Execution Complete ---\n")

     

    # 3. Inspect Results
    print("=" * 70)
    print("1. CONSOLIDATED TEXT (Produced by Intake via STT + OCR):")
    print("=" * 70)
    print(final_state.get("text"))
    print("\n")

    print("=" * 70)
    print("2. AUTONOMOUS SEARCH RESULTS (Produced by Search Agent Loop):")
    print("=" * 70)
    if final_state.get("search_results"):
        parsed_results = json.loads(final_state["search_results"])
        print(json.dumps(parsed_results, ensure_ascii=False, indent=2))
    else:
        print("❌ No search results generated.")