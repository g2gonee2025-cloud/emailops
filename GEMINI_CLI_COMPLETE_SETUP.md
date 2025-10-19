# Gemini Code CLI - Complete Configuration Guide

## Current Status
✅ **Gemini CLI Version**: 0.9.0  
✅ **Installation Path**: `C:\Users\ASUS\AppData\Roaming\npm\gemini.cmd`  
✅ **Configuration Directory**: `C:\Users\ASUS\.gemini`  
⚠️ **Auth Configuration**: Mixed (needs alignment)

## Current Configuration Issues
Your `settings.json` has conflicting authentication settings:
- `auth.type`: "gca" (Google Cloud Auth)
- `security.auth.selectedType`: "vertex-ai"

## Quick Start Commands

### 1. Test Current Setup
```powershell
# Check if Gemini is working
gemini --version

# Start Gemini interactive mode
gemini
```

### 2. Using the Batch Script
```powershell
# Run with your batch script (recommended)
.\run_gemini_cli.bat
```

## Configuration Options

### Authentication Methods

#### Option 1: Google Cloud Auth (GCA) - Recommended for Personal Use
```powershell
# In Gemini interactive mode
> /config auth.type gca
> /auth login
```

#### Option 2: Vertex AI - For Enterprise/Project Use
```powershell
# First, ensure gcloud is configured
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Then in Gemini
> /config auth.type vertex-ai
```

## Optimized Configuration

### Update Settings for Your Project
Create or update `~/.gemini/settings.json`:

```json
{
  "model": {
    "name": {
      "name": "gemini-2.5-pro"
    }
  },
  "general": {
    "preferredEditor": "vscode",
    "autoSave": true,
    "contextWindow": "large"
  },
  "ide": {
    "hasSeenNudge": true,
    "vscodeIntegration": true
  },
  "auth": {
    "type": "gca"
  },
  "approvalMode": "auto_edit",
  "context": {
    "fileFiltering": {
      "respectGitignore": true,
      "excludePatterns": [
        "*.pyc",
        "__pycache__",
        ".git",
        ".venv",
        "*.egg-info",
        "htmlcov",
        ".pytest_cache"
      ]
    },
    "includeDirectories": [
      "c:\\Users\\ASUS\\Downloads\\emailops_vertex_ai"
    ],
    "loadMemoryFromIncludeDirectories": true,
    "maxFileSize": 1048576,
    "maxFiles": 500
  },
  "features": {
    "codeCompletion": true,
    "codeExplanation": true,
    "testGeneration": true,
    "refactoring": true,
    "documentation": true
  }
}
```

## Using Gemini CLI Effectively

### 1. Project Context Setup
```bash
# Start Gemini
gemini

# Set project context (already configured for your project)
> /project c:\Users\ASUS\Downloads\emailops_vertex_ai

# Add specific folders to context
> /folder emailops
> /folder tests
```

### 2. Common Commands

#### File Operations
```bash
# Add file to context
> /file emailops/llm_client.py

# Add multiple files
> /file emailops/processor.py
> /file emailops/email_processing.py
```

#### Code Analysis
```bash
# Analyze code quality
> Analyze the code quality of the current project

# Find issues
> What are the potential bugs in email_processing.py?

# Suggest improvements
> How can I optimize the performance of llm_client.py?
```

#### Code Generation
```bash
# Generate tests
> Generate unit tests for the EmailProcessor class

# Create documentation
> Generate docstrings for all methods in processor.py

# Refactor code
> Refactor the handle_error method to use proper exception handling
```

### 3. Advanced Features

#### Model Selection
```bash
# Switch to different Gemini models
> /model gemini-2.5-pro
> /model gemini-2.5-flash  # Faster, less capable
> /model gemini-2.0-ultra  # Most capable
```

#### Approval Modes
```bash
# Change how edits are handled
> /config approvalMode manual      # Ask before each edit
> /config approvalMode auto_edit   # Auto-apply edits (current)
> /config approvalMode review      # Show diffs before applying
```

#### Context Management
```bash
# View current context
> /context

# Clear context
> /context clear

# Save context for later
> /context save my-session

# Load saved context
> /context load my-session
```

## VS Code Integration

### Install Companion Extension
```bash
# In Gemini CLI
> /ide install

# Or manually in VS Code:
# 1. Press Ctrl+Shift+X
# 2. Search "Gemini CLI Companion"
# 3. Install
```

### VS Code Commands (after extension install)
- `Ctrl+Shift+P` → "Gemini: Ask about selection"
- `Ctrl+Shift+P` → "Gemini: Explain code"
- `Ctrl+Shift+P` → "Gemini: Generate tests"
- `Ctrl+Shift+P` → "Gemini: Refactor"

## PowerShell Aliases

Add to your PowerShell profile (`$PROFILE`):

```powershell
# Quick Gemini start with project context
function Start-GeminiProject {
    param(
        [string]$Query = ""
    )
    
    # Ensure conda environment
    if ($env:CONDA_DEFAULT_ENV -ne "emailops") {
        conda activate emailops
    }
    
    # Start Gemini with optional query
    if ($Query) {
        echo $Query | gemini
    } else {
        gemini
    }
}

# Aliases
Set-Alias -Name gai -Value Start-GeminiProject
Set-Alias -Name gemini-project -Value Start-GeminiProject

# Quick commands
function Gemini-Explain {
    param([string]$File)
    echo "/file $File`nExplain this code in detail" | gemini
}

function Gemini-Test {
    param([string]$File)
    echo "/file $File`nGenerate comprehensive unit tests" | gemini
}

function Gemini-Optimize {
    param([string]$File)
    echo "/file $File`nOptimize this code for performance" | gemini
}
```

## Troubleshooting

### Authentication Issues
```bash
# Clear auth and re-login
> /auth logout
> /auth login

# Check auth status
> /auth status
```

### Context Issues
```bash
# If files aren't being recognized
> /context refresh

# Reset context entirely
> /context reset
```

### Performance Issues
```bash
# Use faster model for quick responses
> /model gemini-2.5-flash

# Reduce context size
> /config context.maxFiles 100
```

## Environment-Specific Setup

### With Conda (Your Current Setup)
```bash
# Activate environment first
conda activate emailops

# Then run Gemini
gemini

# Or use the batch script
.\run_gemini_cli.bat
```

### Global Installation (Alternative)
```bash
# Install globally if not installed
npm install -g @google/gemini-cli

# Run from anywhere
gemini
```

## Best Practices

1. **Keep Context Focused**: Don't add too many files at once
2. **Use Specific Queries**: Be precise about what you want
3. **Leverage History**: Use up/down arrows to recall previous commands
4. **Save Sessions**: Use `/context save` for complex work sessions
5. **Update Regularly**: Check for updates with `npm update -g @google/gemini-cli`

## Quick Test

Run this to verify everything is working:

```bash
gemini

# In interactive mode:
> /config
> /auth status
> /project .
> /file README_PROD.md
> Summarize this project
> /exit
```

## Next Steps

1. ✅ Verify authentication is working
2. ✅ Install VS Code extension for IDE integration
3. ✅ Set up PowerShell aliases for quick access
4. ✅ Test with a simple code analysis task
5. ✅ Configure exclusion patterns for better performance

Your Gemini Code CLI is ready to use! Just run `gemini` or `.\run_gemini_cli.bat` to start.
