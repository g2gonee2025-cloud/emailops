import google.generativeai as genai
import asyncio
import os
import glob

# --- 1. Configuration ---

# IMPORTANT: Set your Google API Key as an environment variable
# In PowerShell: $env:GOOGLE_API_KEY="YOUR_API_KEY"
# In Bash: export GOOGLE_API_KEY="YOUR_API_KEY"
# genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# --- List of files to process ---
# Use glob to easily select multiple files. Examples:
# FILES_TO_PROCESS = glob.glob("src/**/*.py", recursive=True) # All Python files in src
# FILES_TO_PROCESS = ["file1.py", "docs/file2.txt"] # Specific files
FILES_TO_PROCESS = glob.glob("emailops/*.py") # Example: All Python files in the emailops directory

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
MODEL_NAME = "gemini-2.5-pro"
OUTPUT_DIR = "analysis_results"

# --- 2. The Asynchronous API Call Function ---

async def analyze_file(file_path: str):
    """
    Reads a file, formats the prompt, calls the Gemini API, and saves the result.
    """
    print(f"[STARTING] Analysis for {file_path}")
    try:
        # Read the content of the code file
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        # Create the specific prompt for this file
        prompt = MASTER_PROMPT.format(file_content=file_content)

        # Initialize the model and send the request
        model = genai.GenerativeModel(MODEL_NAME)
        response = await model.generate_content_async(prompt)

        # Create the output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Save the response to a new file
        output_filename = os.path.join(OUTPUT_DIR, os.path.basename(file_path) + ".md")
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(response.text)

        print(f"[SUCCESS] Analysis for {file_path} saved to {output_filename}")

    except Exception as e:
        print(f"[FAILED] Analysis for {file_path}. Error: {e}")


# --- 3. The Main Concurrent Runner ---

async def main():
    """
    Creates and runs a list of asynchronous tasks concurrently.
    """
    # Configure the API key from environment variable
    try:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    except KeyError:
        print("FATAL: GOOGLE_API_KEY environment variable not set.")
        return

    # Create a list of tasks to run
    tasks = [analyze_file(file_path) for file_path in FILES_TO_PROCESS]

    # Run all tasks concurrently
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    print("Starting concurrent batch analysis...")
    asyncio.run(main())
    print("All analyses complete.")
