#!/usr/bin/env python3
"""
Simple syntax checker for the vector service implementation
"""
import ast
import sys
import os

def check_syntax(file_path):
    """Check syntax of a Python file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        ast.parse(content)
        print(f"✓ {file_path} - Syntax OK")
        return True
    except SyntaxError as e:
        print(f"✗ {file_path} - Syntax Error: {e}")
        return False
    except Exception as e:
        print(f"✗ {file_path} - Error: {e}")
        return False

def main():
    """Check syntax of all vector service files"""
    files_to_check = [
        'app/services/vector_service.py',
        'app/services/hybrid_search.py',
        'app/services/versioning.py',
        'app/api/endpoints/vectors.py',
        'app/api/endpoints/search.py',
        'app/api/endpoints/versioning.py',
        'app/schemas/vector.py',
        'app/schemas/search.py',
        'app/schemas/versioning.py',
        'app/models/versioning.py',
        'tests/test_vector_service.py',
        'tests/test_hybrid_search.py',
        'tests/test_versioning.py'
    ]
    
    all_good = True
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            if not check_syntax(file_path):
                all_good = False
        else:
            print(f"✗ {file_path} - File not found")
            all_good = False
    
    if all_good:
        print("\n✓ All files have valid syntax!")
    else:
        print("\n✗ Some files have syntax errors")
        sys.exit(1)

if __name__ == "__main__":
    main()