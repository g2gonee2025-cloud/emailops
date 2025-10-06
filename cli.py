#!/usr/bin/env python3
"""
EmailOps Vertex AI - Command Line Interface

Main entry point for all EmailOps operations.
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="EmailOps Vertex AI - Email Processing and Search System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index emails in parallel mode
  %(prog)s index --mode parallel
  
  # Run diagnostics
  %(prog)s diagnose
  
  # Launch web UI
  %(prog)s ui
  
  # Monitor indexing progress
  %(prog)s monitor
  
  # Repair index
  %(prog)s repair
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index email archive')
    index_parser.add_argument('--root', default='.', help='Root directory (default: current)')
    index_parser.add_argument('--mode', choices=['parallel', 'sequential'], 
                            default='parallel', help='Processing mode')
    index_parser.add_argument('--workers', type=int, help='Number of workers for parallel mode')
    index_parser.add_argument('--test-mode', action='store_true', help='Run in test mode')
    index_parser.add_argument('--test-chunks', type=int, default=10, 
                            help='Number of chunks for test mode')
    
    # Chunk command
    chunk_parser = subparsers.add_parser('chunk', help='Chunk email documents')
    chunk_parser.add_argument('--root', default='.', help='Root directory')
    chunk_parser.add_argument('--workers', type=int, help='Number of workers')
    chunk_parser.add_argument('--chunk-size', type=int, default=1600, help='Chunk size')
    chunk_parser.add_argument('--overlap', type=int, default=200, help='Chunk overlap')
    
    # Diagnose command
    diagnose_parser = subparsers.add_parser('diagnose', help='Run diagnostics')
    diagnose_parser.add_argument('--accounts', action='store_true', 
                                help='Test all configured accounts')
    diagnose_parser.add_argument('--index', action='store_true', 
                                help='Verify index alignment')
    diagnose_parser.add_argument('--files', action='store_true', 
                                help='Check all Python files')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor indexing progress')
    monitor_parser.add_argument('--watch', action='store_true', 
                              help='Continuous monitoring')
    
    # Repair command
    repair_parser = subparsers.add_parser('repair', help='Repair index')
    repair_parser.add_argument('--root', default='.', help='Root directory')
    repair_parser.add_argument('--remove-batches', action='store_true', 
                              help='Remove batch files after repair')
    
    # UI command
    ui_parser = subparsers.add_parser('ui', help='Launch web interface')
    ui_parser.add_argument('--port', type=int, default=8501, help='Port number')
    ui_parser.add_argument('--host', default='localhost', help='Host address')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup and configuration')
    setup_parser.add_argument('--enable-apis', action='store_true', 
                            help='Enable Vertex AI APIs')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--live', action='store_true', 
                           help='Run live API tests')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Run analysis')
    analyze_parser.add_argument('--files', action='store_true', 
                              help='Analyze file processing')
    analyze_parser.add_argument('--stats', action='store_true', 
                              help='Show file statistics')
    analyze_parser.add_argument('--chunks', action='store_true', 
                              help='Count chunks')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        # Route to appropriate module based on command
        if args.command == 'index':
            from processing.vertex_indexer import main as index_main
            sys.argv = ['vertex_indexer.py']
            if args.root != '.':
                sys.argv.extend(['--root', args.root])
            sys.argv.extend(['--mode', args.mode])
            if args.workers:
                sys.argv.extend(['--workers', str(args.workers)])
            if args.test_mode:
                sys.argv.append('--test-mode')
                sys.argv.extend(['--test-chunks', str(args.test_chunks)])
            return index_main()
            
        elif args.command == 'chunk':
            from processing.parallel_chunker import main as chunk_main
            os.chdir(args.root)
            return chunk_main()
            
        elif args.command == 'diagnose':
            if args.accounts:
                from diagnostics.diagnose_accounts import main as diagnose_main
                return diagnose_main()
            elif args.index:
                from diagnostics.verify_index_alignment import main as verify_main
                sys.argv = ['verify_index_alignment.py', args.root]
                return verify_main()
            elif args.files:
                from diagnostics.check_all_files import main as check_main
                return check_main()
            else:
                print("Please specify what to diagnose: --accounts, --index, or --files")
                return 1
                
        elif args.command == 'monitor':
            from analysis.monitor_indexing import monitor_worker2
            monitor_worker2()
            return 0
            
        elif args.command == 'repair':
            from processing.repair_vertex_parallel_index import main as repair_main
            sys.argv = ['repair_vertex_parallel_index.py']
            if args.remove_batches:
                sys.argv.append('--remove-batches')
            return repair_main()
            
        elif args.command == 'ui':
            import subprocess
            cmd = ['streamlit', 'run', 'ui/emailops_ui.py', 
                  '--server.port', str(args.port),
                  '--server.address', args.host]
            return subprocess.call(cmd)
            
        elif args.command == 'setup':
            if args.enable_apis:
                from setup.enable_vertex_apis import main as setup_main
                return setup_main()
            else:
                print("Please specify setup action: --enable-apis")
                return 1
                
        elif args.command == 'test':
            if args.live:
                from tests.test_all_accounts_live import main as test_main
                return test_main()
            else:
                print("Please specify test type: --live")
                return 1
                
        elif args.command == 'analyze':
            if args.files:
                from analysis.file_processing_analysis import main
                return main() if hasattr(main, '__call__') else 0
            elif args.stats:
                from analysis.file_stats import main
                return main() if hasattr(main, '__call__') else 0
            elif args.chunks:
                from analysis.count_chunks import count_total_chunks
                count_total_chunks(os.getcwd())
                return 0
            else:
                print("Please specify analysis type: --files, --stats, or --chunks")
                return 1
                
    except ImportError as e:
        print(f"Error: Required module not found - {e}")
        print("Make sure all dependencies are installed and files are properly organized.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())