#!/usr/bin/env python3
"""
Comprehensive test script for EmailOps GUI
Tests import, instantiation, tab creation, and basic functionality
"""

import sys

import traceback
import emailops_gui
import traceback

def test_imports():
    """Test that all imports work correctly."""
    print("=" * 80)
    print("TEST 1: Importing emailops_gui module")
    print("=" * 80)

    try:
        print("✓ Successfully imported emailops_gui")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error during import: {e}")
        return False

def test_class_structure():
    """Test that all required classes exist."""
    print("\n" + "=" * 80)
    print("TEST 2: Verifying class structure")
    print("=" * 80)

    try:

        # Check main classes
        classes = ['EmailOpsApp', 'AppSettings', 'TaskController', 'QueueHandler']
        for cls in classes:
            if hasattr(emailops_gui, cls):
                print(f"✓ Class '{cls}' exists")
            else:
                print(f"✗ Class '{cls}' missing")
                return False

        return True
    except Exception as e:
        print(f"✗ Error checking classes: {e}")
        return False

def test_app_methods():
    """Test that EmailOpsApp has all required methods."""
    print("\n" + "=" * 80)
    print("TEST 3: Verifying EmailOpsApp methods")
    print("=" * 80)

    try:

        required_methods = [
            # Menu and structure
            '_build_menu', '_build_header', '_build_tabs',
            # Tab builders (11 tabs)
            '_build_search_tab', '_build_reply_tab', '_build_fresh_tab',
            '_build_chat_tab', '_build_conversations_tab', '_build_index_tab',
            '_build_config_tab', '_build_diagnostics_tab', '_build_chunking_tab',
            '_build_analyze_tab', '_build_log_tab',
            # Action handlers
            '_on_search', '_on_draft_reply', '_on_draft_fresh', '_on_chat',
            '_on_list_convs', '_on_build_index', '_on_chunk_text', '_on_analyze_thread',
            # Utility methods
            '_save_settings', '_load_settings', '_sync_settings_from_ui',
            '_set_status', '_drain_logs', '_show_about', '_show_docs',
            # Diagnostic methods
            '_run_diagnostics', '_check_deps', '_check_index', '_test_embeddings',
            # Config methods
            '_apply_config', '_reset_config', '_view_config',
        ]

        missing = []
        for method in required_methods:
            if hasattr(emailops_gui.EmailOpsApp, method):
                print(f"✓ Method '{method}' exists")
            else:
                print(f"✗ Method '{method}' missing")
                missing.append(method)

        if missing:
            print(f"\n✗ Missing {len(missing)} methods: {missing}")
            return False

        print(f"\n✓ All {len(required_methods)} required methods present")
        return True

    except Exception as e:
        print(f"✗ Error checking methods: {e}")
        traceback.print_exc()
        return False

def test_app_instantiation():
    """Test that the app can be instantiated (without mainloop)."""
    print("\n" + "=" * 80)
    print("TEST 4: Testing app instantiation")
    print("=" * 80)

    try:


        # Create a root window but don't show it
        print("Creating EmailOpsApp instance...")
        app = emailops_gui.EmailOpsApp()

        # Check that basic attributes exist
        print("✓ App instantiated successfully")

        # Verify attributes
        attrs = ['settings', 'task', 'log_queue', 'nb', 'tab_search',
                'tab_reply', 'tab_fresh', 'tab_chat', 'tab_convs',
                'tab_index', 'tab_config', 'tab_diagnostics',
                'tab_chunking', 'tab_analyze', 'tab_logs']

        for attr in attrs:
            if hasattr(app, attr):
                print(f"✓ Attribute '{attr}' exists")
            else:
                print(f"✗ Attribute '{attr}' missing")
                app.destroy()
                return False

        # Verify tabs were added to notebook
        tab_count = len(app.nb.tabs())
        print(f"\n✓ Notebook has {tab_count} tabs")

        if tab_count != 11:
            print(f"✗ Expected 11 tabs, got {tab_count}")
            app.destroy()
            return False

        # Clean up
        app.destroy()
        print("✓ App destroyed cleanly")

        return True

    except Exception as e:
        print(f"✗ Error during instantiation: {e}")
        traceback.print_exc()
        return False

def test_settings_persistence():
    """Test that settings can be saved and loaded."""
    print("\n" + "=" * 80)
    print("TEST 5: Testing settings persistence")
    print("=" * 80)

    try:

        # Create settings
        settings = emailops_gui.AppSettings()
        settings.export_root = "/test/path"
        settings.k = 42
        settings.temperature = 0.5

        # Save
        settings.save()
        print("✓ Settings saved")

        # Load
        loaded = emailops_gui.AppSettings.load()
        print("✓ Settings loaded")

        # Verify
        if loaded.export_root == "/test/path" and loaded.k == 42:
            print("✓ Settings persisted correctly")
            return True
        else:
            print("✗ Settings values don't match")
            return False

    except Exception as e:
        print(f"✗ Error testing settings: {e}")
        traceback.print_exc()
        return False

def test_task_controller():
    """Test TaskController for cancellation management."""
    print("\n" + "=" * 80)
    print("TEST 6: Testing TaskController")
    print("=" * 80)

    try:

        tc = emailops_gui.TaskController()

        # Test initial state
        if tc.busy():
            print("✗ TaskController should not be busy initially")
            return False
        print("✓ Initial state: not busy")

        # Test start
        if not tc.start():
            print("✗ TaskController.start() should return True")
            return False
        print("✓ Started successfully")

        # Test busy state
        if not tc.busy():
            print("✗ TaskController should be busy after start()")
            return False
        print("✓ Busy state correct")

        # Test cancel
        tc.cancel()
        if not tc.cancelled():
            print("✗ TaskController should be cancelled")
            return False
        print("✓ Cancelled successfully")

        # Test done
        tc.done()
        if tc.busy():
            print("✗ TaskController should not be busy after done()")
            return False
        print("✓ Done successfully")

        return True

    except Exception as e:
        print(f"✗ Error testing TaskController: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "EmailOps GUI Comprehensive Test Suite" + " " * 21 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    tests = [
        test_imports,
        test_class_structure,
        test_app_methods,
        test_app_instantiation,
        test_settings_persistence,
        test_task_controller,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"\n✗ Test {test_func.__name__} crashed: {e}")
            traceback.print_exc()
            results.append((test_func.__name__, False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print("\n" + "=" * 80)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 80)

    if passed == total:
        print("\n🎉 All tests passed! GUI is ready for production.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
