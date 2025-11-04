"""
GUI Constants Module
"""

from pathlib import Path

# --- General ---
APP_NAME = "EmailOps Professional"
ORGANIZATION_NAME = "EmailOps"
ORGANIZATION_DOMAIN = "emailops.ai"
DEFAULT_EXPORT_ROOT = Path.home() / ".emailops"

# --- Analysis Panel ---
ANALYSIS_PANEL_TITLE = "📊 Conversation Analysis"
ANALYSIS_OUTPUT_DIR = DEFAULT_EXPORT_ROOT / "analysis"
ANALYSIS_DEFAULT_TEMP = "0.7"

# --- Chunking Panel ---
CHUNKING_PANEL_TITLE = "🔧 Text Chunking Operations"
CHUNKING_DEFAULT_DESC = "Process only new or changed conversations"

# --- Config Panel ---
CONFIG_PANEL_TITLE = "⚙️ EmailOps Configuration"
CONFIG_DEFAULT_EXPORT_ROOT = str(DEFAULT_EXPORT_ROOT)
CONFIG_DEFAULT_INDEX_DIR = ".email_index"
CONFIG_DEFAULT_PROVIDER = "vertex"
CONFIG_DEFAULT_TEMP = "0.7"
CONFIG_DEFAULT_REPLY_POLICY = "reply_all"
CONFIG_DEFAULT_K = 10
CONFIG_DEFAULT_SIM_THRESHOLD = "0.5"
CONFIG_DEFAULT_MMR_LAMBDA = "0.7"
CONFIG_DEFAULT_RERANK_ALPHA = "0.35"
CONFIG_DEFAULT_CHUNK_SIZE = 2400
CONFIG_DEFAULT_CHUNK_OVERLAP = 240
CONFIG_DEFAULT_NUM_WORKERS = 4
CONFIG_DEFAULT_TARGET_TOKENS = 20000
CONFIG_DEFAULT_EMAIL_CHUNK_LINES = 100

# --- File Panel ---
FILE_PANEL_TITLE = "📁 File Operations"

# --- Main Window ---
MAIN_WINDOW_TITLE = "EmailOps Professional"

# --- Search Panel ---
SEARCH_PANEL_TITLE = "🔍 Intelligent Search"
