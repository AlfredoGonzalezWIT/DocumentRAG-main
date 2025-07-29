# user_query.py

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from faiss_indexing_helper import FAISSIndexingHelper
from embedding_helper import EmbeddingHelper

# Load environment variables
load_dotenv("local.env")

# Azure OpenAI setup
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_BASE = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_DEPLOYMENT_NAME = os.getenv("OPENAI_COMPLETION_MODEL")  # Must be deployment name!

client = OpenAI(
    api_key=AZURE_API_KEY,
    base_url=f"{AZURE_API_BASE}/openai/deployments/{AZURE_DEPLOYMENT_NAME}",
    default_headers={"api-key": AZURE_API_KEY},
    default_query={"api-version": AZURE_API_VERSION}
)

def generate_answer_with_openai_chat(user_query, retrieved_docs):
    try:
        # Prepare message context
        system_prompt = (
            "You are an assistant that answers questions using provided documents. "
            "Use only the relevant information from the documents."
        )

        context = "\n\n".join([doc["content"] for doc in retrieved_docs])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Documents:\n{context}"},
            {"role": "user", "content": user_query}
        ]

        # Make the API call
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.4,
            max_tokens=300
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("❌ Error occurred while generating the response:")
        import traceback
        traceback.print_exc()
        return "❌ I'm sorry, I couldn't generate a response right now."

def main():
    print("🧠 FAISS RAG Chatbot is ready! Type your question or 'exit' to quit.\n")
    
    embedder = EmbeddingHelper()
    searcher = FAISSIndexingHelper()

    while True:
        user_input = input("💬 You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # Step 1: Get embedding
        try:
            query_embedding = embedder.get_embedding(user_input)
        except Exception as e:
            print(f"❌ Failed to generate embedding: {e}")
            continue

        # Step 2: Perform FAISS search
        try:
            top_docs = searcher.search(query_embedding, k=5)
        except Exception as e:
            print(f"❌ Failed to search documents: {e}")
            continue

        if not top_docs:
            print("⚠️ No relevant documents found.")
            continue

        # Step 3: Generate answer from retrieved docs
        answer = generate_answer_with_openai_chat(user_input, top_docs)
        print(f"\n🤖 Answer: {answer}\n")

if __name__ == "__main__":
    main()
