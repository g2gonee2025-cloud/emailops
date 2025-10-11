#!/usr/bin/env python3
"""Test script to verify all 4 critical fixes"""

import sys
import traceback

def test_prepare_index_units():
    """Test 1: Verify prepare_index_units import and usage in email_indexer"""
    try:
        # Test import from text_chunker
        from emailops.text_chunker import prepare_index_units
        print("✅ Test 1a: prepare_index_units imported from text_chunker")
        
        # Test import through email_indexer
        from emailops.email_indexer import prepare_index_units as pi_from_indexer
        print("✅ Test 1b: prepare_index_units accessible from email_indexer")
        
        # Test function execution
        chunks = prepare_index_units(
            'Sample email text for testing purposes',
            'test_id',
            '/test/path',
            'Test Subject',
            '2024-01-01',
            chunk_size=50,
            chunk_overlap=10
        )
        print(f"✅ Test 1c: Function executed, created {len(chunks)} chunks")
        
        # Verify chunk structure
        if chunks and 'id' in chunks[0] and 'text' in chunks[0]:
            print(f"✅ Test 1d: Chunk structure correct, first chunk ID: {chunks[0]['id']}")
        else:
            print("❌ Test 1d: Chunk structure incorrect")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        traceback.print_exc()
        return False

def test_find_conv_ids_by_subject():
    """Test 2: Verify _find_conv_ids_by_subject function"""
    try:
        from emailops.search_and_draft import _find_conv_ids_by_subject
        print("✅ Test 2a: _find_conv_ids_by_subject imported")
        
        # Test with sample data
        test_mapping = [
            {'conv_id': 'conv1', 'subject': 'Invoice for project'},
            {'conv_id': 'conv2', 'subject': 'Meeting notes'},
            {'conv_id': 'conv3', 'subject': 'Another invoice'}
        ]
        
        # Test case-insensitive search
        result = _find_conv_ids_by_subject(test_mapping, 'invoice')
        if result == {'conv1', 'conv3'}:
            print(f"✅ Test 2b: Function works correctly, found {len(result)} conversations")
        else:
            print(f"❌ Test 2b: Unexpected result: {result}")
            return False
        
        # Test empty search
        result_empty = _find_conv_ids_by_subject(test_mapping, '')
        if result_empty == set():
            print("✅ Test 2c: Empty search returns empty set")
        else:
            print(f"❌ Test 2c: Empty search should return empty set, got: {result_empty}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        traceback.print_exc()
        return False

def test_format_analysis_as_markdown():
    """Test 3: Verify format_analysis_as_markdown function"""
    try:
        from emailops.summarize_email_thread import format_analysis_as_markdown
        print("✅ Test 3a: format_analysis_as_markdown imported")
        
        # Test with minimal analysis structure
        test_analysis = {
            'category': 'test',
            'subject': 'Test Email Thread',
            'summary': ['Point 1', 'Point 2', 'Point 3'],
            'participants': [
                {'name': 'John Doe', 'role': 'sender', 'email': 'john@example.com', 'tone': 'professional', 'stance': 'neutral'}
            ],
            'facts_ledger': {
                'explicit_asks': [],
                'commitments_made': [],
                'unknowns': ['Missing information'],
                'forbidden_promises': [],
                'key_dates': []
            },
            'next_actions': [],
            'risk_indicators': ['Risk 1'],
            '_metadata': {
                'analyzed_at': '2024-01-01T00:00:00Z',
                'provider': 'vertex',
                'completeness_score': 85,
                'version': '2.0'
            }
        }
        
        result = format_analysis_as_markdown(test_analysis)
        
        # Verify markdown contains expected sections
        if '# Email Thread Analysis' in result:
            print("✅ Test 3b: Markdown contains header")
        else:
            print("❌ Test 3b: Missing header in markdown")
            return False
        
        if 'Test Email Thread' in result:
            print("✅ Test 3c: Markdown contains subject")
        else:
            print("❌ Test 3c: Missing subject in markdown")
            return False
        
        if 'John Doe' in result:
            print("✅ Test 3d: Markdown contains participant")
        else:
            print("❌ Test 3d: Missing participant in markdown")
            return False
        
        print(f"✅ Test 3e: Generated {len(result)} chars of markdown")
        return True
        
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        traceback.print_exc()
        return False

def test_qdrant_import():
    """Test 4: Verify corrected import path for Qdrant client"""
    try:
        from setup.qdrant_client import QdrantVectorStore, test_qdrant_connection
        print("✅ Test 4a: QdrantVectorStore imported from setup.qdrant_client")
        print("✅ Test 4b: test_qdrant_connection imported from setup.qdrant_client")
        
        # Test that the old import path would fail
        try:
            from emailops.qdrant_client import QdrantVectorStore as OldImport
            print("⚠️  Test 4c: Old import path still works (might have duplicate file)")
        except ImportError:
            print("✅ Test 4c: Old import path correctly fails")
        
        return True
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests and report results"""
    print("=" * 60)
    print("VERIFICATION OF 4 CRITICAL FIXES")
    print("=" * 60)
    
    tests = [
        ("prepare_index_units function", test_prepare_index_units),
        ("_find_conv_ids_by_subject function", test_find_conv_ids_by_subject),
        ("format_analysis_as_markdown function", test_format_analysis_as_markdown),
        ("Qdrant import path", test_qdrant_import)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n📋 Testing: {name}")
        print("-" * 40)
        success = test_func()
        results.append((name, success))
        print()
    
    # Summary
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {name}")
        if not success:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✅ ALL 4 CRITICAL FIXES ARE WORKING CORRECTLY!")
        print("=" * 60)
        return 0
    else:
        print("❌ SOME FIXES FAILED - Review the output above")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())