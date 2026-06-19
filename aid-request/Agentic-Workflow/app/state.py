from typing import TypedDict, Any, Dict, List, Optional, Annotated

class CaseState(TypedDict):
    text: str
    voice_path: Optional[str]
    images: List[str]
    user_id: str
    transcript: Optional[str]
    normalized_case: Dict[str, Any]
    evidence: Dict[str, Any]

    inquiry_history: List[Dict[str, Any]]

    loop_count: int

    reasoning: Dict[str, Any]
    final_output: Dict[str, Any]
    