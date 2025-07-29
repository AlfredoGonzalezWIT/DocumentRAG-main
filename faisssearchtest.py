from embedding_helper import EmbeddingHelper
from faiss_indexing_helper import FAISSIndexingHelper

# Step 1: Initialize helpers
embedder = EmbeddingHelper()
faiss_index = FAISSIndexingHelper()

# Step 2: Search example
query = "What are CRSP responsibilities?"
query_vector = embedder.get_embedding(query)

results = faiss_index.search(query_vector, k=3)

print("\n🔍 Top Matches:")
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print("ID:", doc["id"])
    print("Preview:", doc["content"][:300])
