#!/usr/bin/env python3
"""
Test runner for the charity support system.
Runs individual node and pipeline tests to validate the active inquiry loop.
"""
import sys
import importlib
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def run_test_module(module_name: str, display_name: str) -> bool:
    """
    Dynamically import and run a test module.
    Returns True if successful, False otherwise.
    """
    try:
        print(f"\n{'=' * 60}")
        print(f"Running: {display_name}")
        print(f"{'=' * 60}")
        
        module = importlib.import_module(f"tests.{module_name}")
        
        if hasattr(module, "main"):
            module.main()
            print(f"✓ {display_name} completed successfully")
            return True
        else:
            print(f"✗ {display_name} has no main() function")
            return False
            
    except Exception as e:
        print(f"✗ {display_name} failed with error:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all test modules in sequence."""
    print("\n" + "=" * 60)
    print("CHARITY SUPPORT SYSTEM - TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("test_intake", "Intake Node Tests"),
        ("test_evidence", "Evidence Node Tests"),
        ("test_reasoning", "Reasoning Node Tests"),
        ("test_vqa", "VQA Node Tests"),
        ("test_search", "Search Agent Tests"),
        ("test_report", "Report Node Tests"),
        ("test_pipeline", "Full Pipeline Tests"),
    ]
    
    results = {}
    for module_name, display_name in tests:
        results[display_name] = run_test_module(module_name, display_name)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
