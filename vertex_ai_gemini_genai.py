import os

import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Set up authentication using service account
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Initialize Vertex AI GenAI
project = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_LOCATION")
model_name = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")

# Example: Generate text using Gemini 2.5 Pro

def generate_text(prompt):
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    prompt = "Summarize the following email thread: ..."
    try:
        result = generate_text(prompt)
        print("Gemini Response:", result)
    except Exception as e:
        print("Error during prediction:", e)
