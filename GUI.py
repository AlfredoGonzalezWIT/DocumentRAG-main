import streamlit as st
from embedding_helper import EmbeddingHelper
from faiss_indexing_helper import FAISSIndexingHelper
from user_query import generate_answer_with_openai_chat

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

embedder = EmbeddingHelper()
searcher = FAISSIndexingHelper()

st.title("📄 DocumentRAG Chatbot")

# Display conversation history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Input box BELOW the conversation
user_input = st.chat_input("Ask your question...")

if user_input:
    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        query_embedding = embedder.get_embedding(user_input)
        top_docs = searcher.search(query_embedding, k=5)

        if not top_docs:
            answer = "⚠️ No relevant documents found."
        else:
            answer = generate_answer_with_openai_chat(user_input, top_docs)

        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
