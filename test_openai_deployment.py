from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="9tVpDob9HE1Bp227jVXd3halsscb8bjAKRKlYxYyKRrsJFoKT25dJQQJ99BGACYeBjFXJ3w3AAABACOGF9Kf",
    api_version="2024-12-01-preview",
    azure_endpoint="https://witragopenaitest.openai.azure.com/"
)

response = client.embeddings.create(
    input="This is a test sentence.",
    model="text-embedding-model"  # deployment name, not base model name
)

embedding = response.data[0].embedding
print("Vector length:", len(embedding))
print("First 5 values:", embedding[:5])
print("Embedding length:", len(embedding))