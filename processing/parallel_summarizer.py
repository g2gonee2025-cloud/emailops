#!/usr/bin/env python3
"""
Compatibility shim for parallel_summarizer.py
Redirects to the unified text_processor module
"""

import sys
from text_processor import main

if __name__ == "__main__":
    # Pass through to text_processor with chunk command
    if len(sys.argv) == 1 or sys.argv[1] != "chunk":
        sys.argv.insert(1, "chunk")
    main()
