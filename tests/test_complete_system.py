#!/usr/bin/env python3
"""
Comprehensive system test to validate all components work together
after the augmentation quality improvements.
"""

import subprocess
import sys
from pathlib import Path

def run_test(test_name: str, command: str, expected_keywords: list = None) -> bool:
    """Run a test and check for expected outcomes."""
    
    print(f"\n🧪 Running: {test_name}")
    print("=" * 60)
    
    try:
        # Run the command
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        # Check return code
        if result.returncode != 0:
            print(f"❌ FAILED: {test_name}")
            print(f"Return code: {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return False
        
        # Check for expected keywords if provided
        if expected_keywords:
            output = result.stdout.lower()
            for keyword in expected_keywords:
                if keyword.lower() not in output:
                    print(f"❌ FAILED: {test_name} - Missing expected keyword: {keyword}")
                    return False
        
        print(f"✅ PASSED: {test_name}")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT: {test_name} - Test took longer than 2 minutes")
        return False
    except Exception as e:
        print(f"❌ ERROR: {test_name} - {e}")
        return False

def main():
    """Run comprehensive system tests."""
    
    print("🎯 COMPREHENSIVE SYSTEM TEST SUITE")
    print("=" * 60)
    print("Testing all components after augmentation quality improvements")
    print()
    
    # Define test suite
    tests = [
        {
            "name": "Research-Grade Implementation",
            "command": "python test_research_grade_implementation.py",
            "keywords": ["ALL TESTS PASSED", "RESEARCH-GRADE IMPLEMENTATION READY"]
        },
        {
            "name": "Embedding Cache Efficiency",
            "command": "python test_embedding_cache_efficiency.py",
            "keywords": ["speedup", "cache hit rate", "time saved"]
        },
        {
            "name": "SentenceTransformer Quality",
            "command": "python test_sentence_transformer_quality.py",
            "keywords": ["accuracy", "excellent", "correct ranking"]
        },
        {
            "name": "Global Negative Sampling",
            "command": "python test_global_negative_sampling.py",
            "keywords": ["global negative sampling", "implementation successful"]
        },
        {
            "name": "Normalization Fix",
            "command": "python test_normalization_fix.py",
            "keywords": ["properly normalized", "consistent behavior"]
        },
        {
            "name": "Augmentation Quality Gates",
            "command": "python test_augmentation_quality_gates.py",
            "keywords": ["ALL TESTS PASSED", "AUGMENTATION QUALITY IMPROVEMENTS VALIDATED"]
        },
        {
            "name": "Loss Function Validation",
            "command": "python run_loss_validation.py --synthetic",
            "keywords": ["validation completed successfully", "critical issues: 0"]
        },
        {
            "name": "Small Training Test",
            "command": "python run_small_training_test.py",
            "keywords": ["SUCCESS", "Trainer is compatible"]
        }
    ]
    
    # Run all tests
    results = []
    
    for test in tests:
        success = run_test(
            test["name"],
            test["command"],
            test.get("keywords", [])
        )
        results.append((test["name"], success))
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 COMPREHENSIVE TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall Results: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("\n🎉 ALL SYSTEM TESTS PASSED!")
        print("\n✅ System Status: READY FOR PRODUCTION")
        print("\n🚀 Key Capabilities Validated:")
        print("  • Research-grade contrastive learning implementation")
        print("  • Embedding collapse prevention through quality gates")
        print("  • High-performance caching and optimization")
        print("  • Robust loss function computation")
        print("  • Career-aware data augmentation with quality controls")
        print("  • Global negative sampling for better training")
        print("  • Proper normalization and numerical stability")
        print("  • End-to-end training pipeline functionality")
        
        print("\n📊 System Ready For:")
        print("  • Large-scale training on full dataset")
        print("  • Research paper experiments and validation")
        print("  • Production deployment for resume-job matching")
        print("  • Comparative studies against baseline methods")
        
    else:
        failed_tests = [name for name, success in results if not success]
        print(f"\n⚠️ {total - passed} tests failed:")
        for test_name in failed_tests:
            print(f"  • {test_name}")
        print("\n🔧 Action Required: Review failed tests before production use")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)