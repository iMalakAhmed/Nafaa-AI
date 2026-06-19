#!/usr/bin/env python3
"""
Quick test runner - Execute this from the project root:
    python quick_test.py
"""
import sys
import os

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import test modules
from tests import test_reasoning, test_vqa, test_search, test_pipeline, test_report

print("""
╔════════════════════════════════════════════════════════════════╗
║    CHARITY SUPPORT SYSTEM - QUICK TEST                         ║
║    Testing the Active Inquiry Loop Architecture                ║
╚════════════════════════════════════════════════════════════════╝
""")

tests = [
    ("Reasoning Node (Decision Making)", test_reasoning.main),
    ("VQA Node (Image Analysis)", test_vqa.main),
    ("Search Agent (Information Retrieval)", test_search.main),
    ("Report Node (Final Decision)", test_report.main),
    ("Full Pipeline (End-to-End)", test_pipeline.main),
]

passed = 0
failed = 0

for name, test_func in tests:
    try:
        print(f"\n▶ {name}")
        print("─" * 60)
        test_func()
        passed += 1
        print(f"✓ {name} PASSED")
    except Exception as e:
        failed += 1
        print(f"✗ {name} FAILED: {e}")

print(f"""
╔════════════════════════════════════════════════════════════════╗
║    RESULTS: {passed} passed, {failed} failed                        
╚════════════════════════════════════════════════════════════════╝
""")

sys.exit(0 if failed == 0 else 1)
