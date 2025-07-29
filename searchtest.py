import requests
from embedding_helper import EmbeddingHelper

# Get embedding for query
embedder = EmbeddingHelper()
embedding_vector = embedder.get_embedding("Find documents about CRSP responsibilities.")

# Azure Search setup
url = "https://witragtest.search.windows.net/indexes/knowledge-index/docs/search?api-version=2023-10-01-Preview"
headers = {
    "Content-Type": "application/json",
    "api-key": "Hw8Ai0XbpA4eSrlZA1TpDolwgjIksHKr787sdIOvy1AzSeBhtoCK"  # Replace with os.getenv() if stored in .env
}

# Search payload
payload = {
    "vectorQueries": [
        {
            "kind": "vector",
            "vector": embedding_vector,
            "fields": "contentVector",
            "k": 3,
            "profile": "hnswProfile"
        }
    ]
}

# Send search request
response = requests.post(url, headers=headers, json=payload)

# ✅ Check response before trying .json()
print(f"Status Code: {response.status_code}")

if response.ok:
    print("✅ Search response:")
    print(response.json())
else:
    print("❌ Error response:")
    print(response.text)
