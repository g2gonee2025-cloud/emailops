# EmailOps Vertex AI

A comprehensive email indexing and search system powered by Google Vertex AI, designed for processing and searching through large email archives with AI-powered capabilities.

## 🚀 Features

- **Email Processing**: Efficiently process and index email archives from Outlook exports
- **AI-Powered Search**: Semantic search using Vertex AI embeddings
- **Email Drafting**: AI-assisted email composition based on historical context
- **Parallel Processing**: Multi-worker architecture for fast indexing
- **Web UI**: Streamlit-based interface for easy interaction
- **Multi-Account Support**: Process emails from multiple Google Cloud accounts

## 📁 Project Structure

```
emailops_vertex_ai/
│
├── emailops/                 # Core library modules
│   ├── doctor.py            # System diagnostics
│   ├── email_indexer.py     # Email indexing logic
│   ├── env_utils.py         # Environment utilities
│   ├── index_metadata.py    # Index metadata management
│   ├── llm_client.py        # LLM client implementations
│   ├── search_and_draft.py  # Search and email drafting
│   ├── summarize_email_thread.py  # Email thread summarization
│   ├── text_chunker.py      # Text chunking utilities
│   └── utils.py             # General utilities
│
├── diagnostics/              # Diagnostic and debugging tools
│   ├── diagnose_accounts.py # Account diagnostics
│   ├── debug_parallel_indexer.py
│   ├── check_failed_batches.py
│   ├── verify_index_alignment.py
│   └── check_all_files.py
│
├── processing/               # Data processing scripts
│   ├── vertex_indexer.py    # Main indexing script
│   ├── parallel_chunker.py  # Parallel text chunking
│   ├── parallel_summarizer.py
│   ├── fix_failed_embeddings.py
│   ├── repair_vertex_parallel_index.py
│   └── run_vertex_finalize.py
│
├── analysis/                 # Analysis and statistics tools
│   ├── file_processing_analysis.py
│   ├── file_stats.py
│   ├── count_chunks.py
│   └── monitor_indexing.py
│
├── tests/                    # Test scripts
│   └── test_all_accounts_live.py
│
├── setup/                    # Setup and configuration
│   ├── enable_vertex_apis.py
│   ├── setup_vertex_env.bat
│   ├── activate_env.bat
│   └── activate_env.ps1
│
├── utils/                    # Utility modules
│   └── vertex_utils.py
│
├── data/                     # Data files
│   ├── validated_accounts.json
│   ├── account_diagnostics.json
│   └── live_api_test_results.json
│
├── docs/                     # Documentation
│   └── WORKER_ISSUE_REPORT.md
│
├── ui/                       # User interface
│   └── emailops_ui.py
│
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container configuration
├── environment.yml          # Conda environment
└── .env.example            # Environment variables template
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Google Cloud Account with Vertex AI enabled
- Conda (recommended) or pip

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd emailops_vertex_ai
   ```

2. **Create environment**
   ```bash
   # Using Conda (recommended)
   conda env create -f environment.yml
   conda activate emailops

   # Or using pip
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Enable Vertex AI APIs**
   ```bash
   python setup/enable_vertex_apis.py
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
VERTEX_LOCATION=us-central1
VERTEX_EMBED_MODEL=textembedding-gecko@003

# Email Export Path
EXPORT_ROOT=C:/Users/ASUS/Desktop/Outlook

# Optional: OpenAI Configuration
OPENAI_API_KEY=your-openai-key
```

### Account Configuration

Configure multiple Google Cloud accounts in `data/validated_accounts.json`:

```json
{
  "accounts": [
    {
      "project_id": "project-1",
      "region": "us-central1",
      "credentials_path": "path/to/credentials.json"
    }
  ]
}
```

## 📖 Usage

### 1. Process Email Archive

```bash
# Run the main indexer
python processing/vertex_indexer.py --root . --mode parallel

# Or use sequential mode for debugging
python processing/vertex_indexer.py --root . --mode sequential
```

### 2. Launch Web UI

```bash
streamlit run ui/emailops_ui.py
```

### 3. Monitor Progress

```bash
python analysis/monitor_indexing.py
```

### 4. Diagnose Issues

```bash
# Check account configuration
python diagnostics/diagnose_accounts.py

# Verify index alignment
python diagnostics/verify_index_alignment.py
```

## 📊 Processing Pipeline

1. **Chunking**: Splits emails and attachments into processable chunks
2. **Embedding**: Generates vector embeddings using Vertex AI
3. **Indexing**: Creates FAISS index for similarity search
4. **Search**: Performs semantic search on indexed content
5. **Drafting**: Generates email responses using LLM

## 🚦 Monitoring

The system provides comprehensive monitoring:

- Real-time progress tracking
- Worker status monitoring
- Error logging and diagnostics
- Performance metrics

## 🐛 Troubleshooting

### Common Issues

1. **Worker failures**: Check `logs/` directory for error details
2. **API errors**: Verify credentials and API enablement
3. **Memory issues**: Reduce batch size or number of workers
4. **Index corruption**: Use `processing/repair_vertex_parallel_index.py`

### Diagnostic Tools

- `diagnostics/diagnose_accounts.py`: Test account configuration
- `diagnostics/check_failed_batches.py`: Identify failed batches
- `diagnostics/verify_index_alignment.py`: Check index integrity

## 📝 Development

### Running Tests

```bash
python tests/test_all_accounts_live.py
```

### Code Quality

```bash
python diagnostics/check_all_files.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

[License information here]

## 🙏 Acknowledgments

- Google Vertex AI team
- Streamlit community
- FAISS developers

## 📧 Contact

[Contact information here]