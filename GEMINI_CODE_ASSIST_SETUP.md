# Gemini Code Assist CLI with Conda Environment

## Current Setup Status ✅

The Gemini Code Assist CLI (`@google/gemini-cli` v0.9.0) is **already working** with your conda environment!

### Installation Details
- **Installed via**: npm (globally)
- **Location**: `C:\Users\ASUS\AppData\Roaming\npm\`
- **Version**: 0.9.0
- **Conda Environment**: emailops (currently active)

## How to Use

### Method 1: Using the Batch Script (Recommended)
```bash
# This automatically handles conda environment and PATH setup
.\run_gemini_cli.bat

# Check version
.\run_gemini_cli.bat --version

# Get help
.\run_gemini_cli.bat --help
```

### Method 2: Manual Setup
```bash
# 1. Activate conda environment
conda activate emailops

# 2. Add npm to PATH (temporary)
$env:PATH += ";C:\Users\ASUS\AppData\Roaming\npm"

# 3. Run Gemini
C:\Users\ASUS\AppData\Roaming\npm\gemini.cmd

# Or use the full path directly
C:\Users\ASUS\AppData\Roaming\npm\gemini.cmd --version
```

## Available Commands in Gemini Code Assist

Once you start `gemini`, you can use these commands:

### Basic Commands
- `/help` - Show available commands
- `/clear` - Clear the conversation
- `/exit` or `/quit` - Exit the CLI

### IDE Integration
- `/ide install` - Install VS Code companion extension
- `/ide uninstall` - Uninstall VS Code companion extension

### Project Context
- `/project` - Set or view project context
- `/file <path>` - Add a file to context
- `/folder <path>` - Add a folder to context

### Configuration
- `/config` - View or modify configuration
- `/model` - Switch between Gemini models
- `/sandbox` - Toggle sandbox mode

## Working with Your Project

### Add Project Context
```bash
# In Gemini interactive mode
> /project c:\Users\ASUS\Downloads\emailops_vertex_ai
> /folder emailops
```

### Ask Questions About Your Code
```bash
> What does the email_processing.py file do?
> How can I improve the error handling in this project?
> Generate unit tests for the llm_client module
```

## Troubleshooting

### If `gemini` command not found:
1. Ensure npm global bin is in PATH:
   ```bash
   $env:PATH += ";C:\Users\ASUS\AppData\Roaming\npm"
   ```

2. Or use full path:
   ```bash
   C:\Users\ASUS\AppData\Roaming\npm\gemini.cmd
   ```

### VS Code Extension Installation Failed
If `/ide install` fails, manually install:
1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "Gemini CLI Companion"
4. Install manually

## Integration with Conda Workflow

### Create an Alias (Optional)
Add to your PowerShell profile:
```powershell
# Add to $PROFILE
function Start-Gemini {
    if ($env:CONDA_DEFAULT_ENV -ne "emailops") {
        conda activate emailops
    }
    gemini
}
Set-Alias -Name gai -Value Start-Gemini
```

Then use:
```bash
gai  # Starts Gemini with emailops environment
```

## Key Differences from gemini-vertex CLI

| Feature | Gemini Code Assist (`gemini`) | Vertex AI CLI (`gemini-vertex`) |
|---------|-------------------------------|----------------------------------|
| Purpose | Interactive AI coding assistant | Command-line Gemini API access |
| Auth | Google account login | Service account (Vertex AI) |
| UI | Interactive TUI with ASCII art | Simple CLI tool |
| Context | Project-aware, file context | Single prompts/conversations |
| IDE | VS Code integration | No IDE integration |

## Best Practices

1. **Always activate conda environment first**:
   ```bash
   conda activate emailops
   gemini
   ```

2. **Set project context** when starting:
   ```bash
   > /project .
   ```

3. **Use specific file context** for targeted help:
   ```bash
   > /file emailops/llm_client.py
   > How can I optimize this code?
   ```

4. **Save useful responses** with output redirection:
   ```bash
   gemini > output.txt
   ```

## Current Status
✅ Gemini Code Assist CLI is working in your conda environment
✅ You can run it directly with the `gemini` command
✅ The emailops conda environment is currently active

Just type `gemini` to start using it!
