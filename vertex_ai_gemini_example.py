from google.cloud import aiplatform

# Initialize Vertex AI with your project and location
# Make sure GOOGLE_APPLICATION_CREDENTIALS is set in your .env or environment

aiplatform.init(
    project="boxwood-well-475205-d4",
    location="global"
)

# Example: Generate text using Gemini 2.5 Pro model
# You must have access to the model and the right IAM permissions

def generate_text(prompt):
    model = aiplatform.Model(model_name="publishers/google/models/gemini-2.5-pro")
    response = model.predict(instances=[{"content": prompt}])
    return response

if __name__ == "__main__":
    prompt = "Summarize the following email thread: ..."
    try:
        result = generate_text(prompt)
        print("Gemini Response:", result)
    except Exception as e:
        print("Error during prediction:", e)
