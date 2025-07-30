import streamlit as st
from embedding_helper import EmbeddingHelper
from faiss_indexing_helper import FAISSIndexingHelper
from user_query import generate_answer_with_openai_chat

# Initialize helpers
embedder = EmbeddingHelper()
searcher = FAISSIndexingHelper()

# App title
st.title("📄 DocumentRAG Chatbot")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input form (clears after submission)
with st.form("question_form", clear_on_submit=True):
    user_input = st.text_input("Ask your question:")
    submitted = st.form_submit_button("Submit")

# Handle input
if submitted and user_input:
    with st.spinner("Thinking..."):
        query_embedding = embedder.get_embedding(user_input)
        top_docs = searcher.search(query_embedding, k=5)

        if not top_docs:
            answer = "⚠️ No relevant documents found."
        else:
            answer = generate_answer_with_openai_chat(user_input, top_docs)

        # Save to history
        st.session_state.chat_history.append((user_input, answer))

# Display full conversation
if st.session_state.chat_history:
    st.subheader("🧠 Conversation History")
    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Assistant:** {a}")
        if i < len(st.session_state.chat_history) - 1:
            st.markdown("---")
