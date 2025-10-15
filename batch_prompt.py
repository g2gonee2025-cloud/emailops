import asyncio
import os
from pathlib import Path

from google import genai

# --- 1. Configuration ---
# Set your Vertex AI project and location via environment variables or directly here
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

# --- List of files to process ---
# Use glob to easily select multiple files. Examples:
# FILES_TO_PROCESS = Path("src").rglob("*.py") # All Python files in src
# FILES_TO_PROCESS = [Path("file1.py"), Path("docs/file2.txt")] # Specific files
FILES_TO_PROCESS = Path("emailops").glob("*.py")  # Example: All Python files in the emailops directory

# --- The Master Prompt ---
# The {file_content} placeholder will be replaced with the content of each file.
MASTER_PROMPT = """
As a Senior Software Engineer, please review the following code for quality, potential bugs, and adherence to best practices. Provide a detailed analysis and suggest specific improvements.

Here is the code:
---
{file_content}
---
"""

# --- Model Configuration ---
MODEL_NAME = "gemini-2.5-pro"  # For text generation
EMBED_MODEL = "gemini-embedding-001"  # For embeddings
OUTPUT_DIR = "analysis_results"

# --- 2. The Asynchronous API Call Function ---

async def analyze_file(file_path: Path, client):
    print(f"[STARTING] Analysis for {file_path}")
    try:
        file_content = file_path.read_text(encoding='utf-8')
        prompt = MASTER_PROMPT.format(file_content=file_content)

        # Generate content using Gemini 2.5 Pro
        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        output_filename = output_dir / f"{file_path.stem}.md"
        output_filename.write_text(response.text, encoding='utf-8')
        print(f"[SUCCESS] Analysis for {file_path} saved to {output_filename}")
    except Exception as e:
        print(f"[FAILED] Analysis for {file_path}. Error: {e}")

# Example embedding usage
async def embed_file(file_path: Path, client):
    try:
        file_content = file_path.read_text(encoding='utf-8')
        response = await client.aio.models.embed_content(
            model=EMBED_MODEL,
            contents=file_content
        )
        print(f"Embedding for {file_path}: {response.embeddings[0].values[:5]}")
    except Exception as e:
        print(f"[FAILED] Embedding for {file_path}. Error: {e}")

async def main():
    # Create Vertex AI client
    client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

    tasks = [analyze_file(file_path, client) for file_path in FILES_TO_PROCESS]
    await asyncio.gather(*tasks)

    # Optionally, run embeddings for each file
    # embed_tasks = [embed_file(file_path, client) for file_path in FILES_TO_PROCESS]
    # await asyncio.gather(*embed_tasks)

if __name__ == "__main__":
    print("Starting concurrent batch analysis...")
    asyncio.run(main())
    print("All analyses complete.")
