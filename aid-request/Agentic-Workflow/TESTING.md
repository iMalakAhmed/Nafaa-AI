# Testing Guide

## Quick Start

Run all tests at once:

```bash
python run_tests.py
```

## Individual Tests

Run a specific test module:

```bash
# Test the intake node (processes text, audio, images)
python -m pytest tests/test_intake.py -v

# Test the evidence node (fraud detection, quality checks)
python -m pytest tests/test_evidence.py -v

# Test the reasoning node (decision logic for VQA/Search/Report)
python -m pytest tests/test_reasoning.py -v

# Test VQA node (image understanding)
python -m pytest tests/test_vqa.py -v

# Test search agent (information retrieval)
python -m pytest tests/test_search.py -v

# Test report node (final decision output)
python -m pytest tests/test_report.py -v

# Test full pipeline (end-to-end workflow)
python -m pytest tests/test_pipeline.py -v
```

## What Each Test Does

### test_intake.py
- **Text only**: Processes plain text input
- **Audio only**: Transcribes audio and extracts text
- **Images only**: Processes image files
- **Combined**: Handles multiple input types together

### test_evidence.py
- Validates cold acquisition evidence
- Checks fraud detection scoring
- Verifies image quality and duplication detection

### test_reasoning.py (NEW LOGIC)
Tests the simplified decision-making:
- **Images present**: Recommends VQA to analyze them
- **Drug name known**: Recommends Search to find pricing
- **Clear case**: Recommends Report for final decision

Expected output format:
```json
{
  "next_step": "vqa | search | report",
  "action_details": {
    "target": "The question or query to execute",
    "reasoning": "Why this step was chosen"
  }
}
```

### test_vqa.py
- Single image with default questions
- Multiple images with specific question
- No images (should return empty)

### test_search.py
- Drug information search
- Medical case search with alternatives

### test_report.py
Tests final reporting:
- Report with clear reasoning
- Report with search results included

### test_pipeline.py
Simulates the full workflow:
1. Intake (gather inputs)
2. Evidence (validate data)
3. Reasoning (decide next step)
4. VQA or Search (if needed)
5. Report (generate final decision)

## Expected Output

Each test prints results in JSON format showing:
- Input state
- Node output
- Decision pathway

Example:
```
=== Reasoning: Images present -> use VQA ===
{
  "reasoning": {
    "next_step": "vqa",
    "action_details": {
      "target": "ما محتوى الصورة بالتفصيل؟",
      "reasoning": "هناك صور مرفقة بحاجة لتحليل."
    }
  }
}
```

## Troubleshooting

### Missing dependencies
If you get import errors, install required packages:
```bash
pip install -r requirements.txt
```

### File not found errors
Tests create dummy files in the `data/` directory. Ensure it exists:
```bash
mkdir -p data
```

### Module import errors
Make sure you're running tests from the project root directory:
```bash
cd "e:\NU\Grad\Coding\Requests V2"
python run_tests.py
```

## Test Architecture

All tests use mocked services to avoid dependencies:
- `llm_model.invoke()` → Returns pre-defined JSON responses
- `answer_three_questions_batch()` → Returns mock VQA results
- `transcribe()` → Returns mock transcript
- `extract_text_from_image()` → Returns mock OCR text
- etc.

This allows testing without:
- Ollama service running
- Real LLM API calls
- External APIs (Tavily, Reverse Image, etc.)

## Adding New Tests

1. Create a new file: `tests/test_my_node.py`
2. Import test utilities and mock services
3. Define a `main()` function that runs test cases
4. Add test module to `run_tests.py`

Example template:

```python
import importlib
from types import SimpleNamespace

def _mock_my_service(input_data):
    return {"result": "mock response"}

def import_my_module():
    module = importlib.import_module("app.nodes.my_node")
    return module

def run_case(case_name, state):
    module = import_my_module()
    result = module.my_node(state)
    print(f"\n=== {case_name} ===")
    print(result)

def main():
    cases = [
        {"name": "Test case 1", "state": {...}},
        {"name": "Test case 2", "state": {...}},
    ]
    for case in cases:
        run_case(case["name"], case["state"])

if __name__ == "__main__":
    main()
```
