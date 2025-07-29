import streamlit as st
from embedding_helper import EmbeddingHelper
from faiss_indexing_helper import FAISSIndexingHelper
from user_query import generate_answer_with_openai_chat

embedder = EmbeddingHelper()
searcher = FAISSIndexingHelper()

st.title("📄 DocumentRAG Chatbot")
user_input = st.text_input("Ask your question:")

if user_input:
    with st.spinner("Thinking..."):
        query_embedding = embedder.get_embedding(user_input)
        top_docs = searcher.search(query_embedding, k=5)
        if not top_docs:
            st.warning("No relevant documents found.")
        else:
            answer = generate_answer_with_openai_chat(user_input, top_docs)
            st.success(answer)
