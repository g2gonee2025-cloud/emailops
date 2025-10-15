#!/usr/bin/env python3
"""
Test script to verify GenAI authentication setup
"""
import os
import sys

from dotenv import load_dotenv


def test_genai_authentication():
    """Test GenAI authentication and basic functionality."""

    # Load environment variables
    load_dotenv()

    print("=== GenAI Authentication Test ===")
    print()

    # Check environment variables
    print("1. Checking environment variables...")
    required_vars = [
        'GOOGLE_CLOUD_PROJECT',
        'GOOGLE_CLOUD_LOCATION',
        'GOOGLE_APPLICATION_CREDENTIALS',
        'GOOGLE_GENAI_USE_VERTEXAI'
    ]

    for var in required_vars:
        value = os.getenv(var)
        if value:
            if var == 'GOOGLE_APPLICATION_CREDENTIALS':
                # Check if credentials file exists
                if os.path.exists(value):
                    print(f"   ✓ {var}: {value} (file exists)")
                else:
                    print(f"   ✗ {var}: {value} (file NOT found)")
                    return False
            else:
                print(f"   ✓ {var}: {value}")
        else:
            print(f"   ✗ {var}: NOT SET")
            return False

    print()

    # Test GenAI import and initialization
    print("2. Testing GenAI import...")
    try:
        import google.generativeai as genai
        print("   ✓ GenAI module imported successfully")
    except ImportError as e:
        print(f"   ✗ Failed to import GenAI: {e}")
        return False

    print()

    # Test Vertex AI configuration
    print("3. Testing Vertex AI configuration...")
    try:
        # Configure GenAI to use Vertex AI
        if os.getenv('GOOGLE_GENAI_USE_VERTEXAI', '').lower() == 'true':
            genai.configure(
                api_key=None,  # Uses service account
                transport='grpc'
            )
            print("   ✓ GenAI configured for Vertex AI")
        else:
            print("   ! GOOGLE_GENAI_USE_VERTEXAI not set to True")
    except Exception as e:
        print(f"   ✗ Failed to configure GenAI: {e}")
        return False

    print()

    # Test model listing
    print("4. Testing model access...")
    try:
        models = list(genai.list_models())
        if models:
            print(f"   ✓ Successfully accessed {len(models)} models")
            # Show first few models
            for i, model in enumerate(models[:3]):
                print(f"      - {model.name}")
        else:
            print("   ! No models found (this might be normal)")
    except Exception as e:
        print(f"   ✗ Failed to list models: {e}")
        print("   Note: This might be due to permissions or API configuration")

    print()

    # Test simple generation (if possible)
    print("5. Testing text generation...")
    try:
        model_name = os.getenv('VERTEX_MODEL', 'gemini-2.5-pro')
        model = genai.GenerativeModel(model_name)

        response = model.generate_content(
            "Hello! Can you confirm you're working? Please respond with just 'Authentication successful!'"
        )

        if response and response.text:
            print(f"   ✓ Generation successful: {response.text.strip()}")
        else:
            print("   ✗ Generation returned empty response")
            return False

    except Exception as e:
        print(f"   ✗ Failed to generate content: {e}")
        return False

    print()
    print("🎉 All tests passed! GenAI authentication is working correctly.")
    return True

if __name__ == "__main__":
    success = test_genai_authentication()
    sys.exit(0 if success else 1)
