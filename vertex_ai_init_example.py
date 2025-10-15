from google.cloud import aiplatform

aiplatform.init(
    project="boxwood-well-475205-d4",
    location="global"
)

# Example: List all models in the project
models = aiplatform.Model.list()
for model in models:
    print(model.resource_name)
