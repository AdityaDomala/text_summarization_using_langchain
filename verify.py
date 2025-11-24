#!/usr/bin/env python3
"""
Quick verification script to test the text summarizer implementation
without requiring API credentials.
"""

import sys
import os


def test_file_structure():
    """Verify all required files exist."""
    print("Testing file structure...")
    
    required_files = [
        'text_summarizer.py',
        'examples.py',
        'requirements.txt',
        '.env.example',
        '.gitignore',
        'README.md'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file} exists")
        else:
            print(f"  ✗ {file} missing")
            return False
    
    return True


def test_syntax():
    """Verify Python files have valid syntax."""
    print("\nTesting Python syntax...")
    
    python_files = ['text_summarizer.py', 'examples.py']
    
    for file in python_files:
        try:
            with open(file, 'r') as f:
                compile(f.read(), file, 'exec')
            print(f"  ✓ {file} has valid syntax")
        except SyntaxError as e:
            print(f"  ✗ {file} has syntax error: {e}")
            return False
    
    return True


def test_requirements():
    """Verify requirements.txt has expected dependencies."""
    print("\nTesting requirements...")
    
    with open('requirements.txt', 'r') as f:
        content = f.read()
    
    required_deps = ['langchain', 'openai', 'python-dotenv', 'tiktoken']
    
    for dep in required_deps:
        if dep in content:
            print(f"  ✓ {dep} in requirements.txt")
        else:
            print(f"  ✗ {dep} missing from requirements.txt")
            return False
    
    return True


def test_readme():
    """Verify README has proper content."""
    print("\nTesting README...")
    
    with open('README.md', 'r') as f:
        content = f.read()
    
    required_sections = [
        'Installation',
        'Usage',
        'Features',
        'Requirements'
    ]
    
    for section in required_sections:
        if section in content:
            print(f"  ✓ '{section}' section present")
        else:
            print(f"  ✗ '{section}' section missing")
            return False
    
    return True


def test_env_example():
    """Verify .env.example exists and contains required keys."""
    print("\nTesting .env.example...")
    
    with open('.env.example', 'r') as f:
        content = f.read()
    
    if 'OPENAI_API_KEY' in content:
        print("  ✓ OPENAI_API_KEY present in .env.example")
        return True
    else:
        print("  ✗ OPENAI_API_KEY missing from .env.example")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Text Summarization Implementation Verification")
    print("=" * 60)
    
    tests = [
        test_file_structure,
        test_syntax,
        test_requirements,
        test_readme,
        test_env_example
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✓ All verification tests passed!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Create .env file: cp .env.example .env")
        print("3. Add your OpenAI API key to .env")
        print("4. Run examples: python examples.py")
        print("=" * 60)
        return 0
    else:
        print("✗ Some tests failed!")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
