#!/usr/bin/env python3
"""
Quick GenAI authentication test
"""
import os
import sys

from dotenv import load_dotenv


def quick_test():
    # Load environment variables
    load_dotenv()

    print("Testing GenAI authentication...")
    print()

    try:
        import google.generativeai as genai
        print("✓ GenAI imported successfully")

        # Set up authentication
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        model_name = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")

        print(f"✓ Using model: {model_name}")
        print(f"✓ Project: {os.getenv('GOOGLE_CLOUD_PROJECT')}")
        print()

        # Quick test
        print("Sending test prompt...")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Say 'Hello, authentication works!'")

        if response and response.text:
            print(f"✓ SUCCESS: {response.text.strip()}")
            return True
        else:
            print("✗ FAILED: No response received")
            return False

    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    print()
    if success:
        print("🎉 GenAI authentication is working!")
    else:
        print("❌ GenAI authentication failed")
    sys.exit(0 if success else 1)
