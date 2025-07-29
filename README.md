# DocumentRAG: Retrieval-Augmented Generation with FAISS and Azure OpenAI

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** chatbot that:

* Extracts and embeds text from PDF documents
* Indexes document chunks with **FAISS**
* Uses **Azure OpenAI** to answer user queries based on retrieved document chunks

---

## 🚀 Features

* PDF ingestion and text chunking
* Embedding generation using Azure OpenAI
* Similarity search with FAISS vector index
* OpenAI chat-based response generation
* Command-line chatbot interface

---

## 🛠️ Requirements

* Python 3.9+
* Azure OpenAI resource with deployed embedding and chat models
* FAISS library

### 🔧 Installation

```bash
# Clone the repo
git clone https://github.com/AlfredoGonzalezWIT/DocumentRAG-main.git
cd DocumentRAG-main

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Example .env File (`local.env`)

```
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_ENGINE=text-embedding-model-deployment-name
OPENAI_COMPLETION_MODEL=gpt-4-chat-deployment-name
AZURE_OPENAI_API_VERSION=2024-12-01-preview
CURRENT_USER_ID=your-email@example.com
```

---

## 📁 Folder Structure

```
DocumentRAG-main/
├── RAG/
│   ├── pdfs/                   # PDF documents to ingest
│   ├── main.log                # Runtime logs
│   └── ...
├── main.py                    # Indexes PDF documents
├── user_query.py              # CLI chatbot
├── embedding_helper.py        # Embedding using Azure OpenAI
├── faiss_indexing_helper.py   # FAISS vector index management
├── local.env                  # Environment variables
├── vector.index               # FAISS index file
├── documents.pkl              # Metadata for indexed chunks
```

---

## 📥 Step-by-Step Tutorial

### 1. Prepare Your Azure OpenAI Deployment

* Deploy an **embedding model** (e.g. `text-embedding-3-large`)
* Deploy a **chat model** (e.g. `gpt-4` or `gpt-35-turbo`)
* Copy the deployment names and fill in your `local.env`

### 2. Add PDF Files

Place PDF documents inside the `RAG/pdfs/` folder.

### 3. Run the Indexer

```bash
python main.py
```

This will:

* Extract text from each PDF
* Chunk it into \~3000 character blocks
* Generate embeddings for each chunk
* Save vectors and metadata into FAISS index

### 4. Start the Chatbot

```bash
python user_query.py
```

You can now ask questions like:

```
What are the responsibilities of CRPS?
Summarize the key terms from the onboarding policy.
```

The assistant will:

* Embed your query
* Perform a similarity search over FAISS index
* Use top results as context
* Send to OpenAI chat completion

---

## 🧼 Resetting the Index

To re-index documents from scratch:

```bash
rm vector.index documents.pkl
python main.py
```

---

## 📚 Credits

* [FAISS](https://github.com/facebookresearch/faiss)
* [Azure OpenAI](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/)

---

## 📌 Next Steps

* Add web or GUI interface
* Enable support for other file types (DOCX, TXT)
* Use LangChain or LlamaIndex for more flexibility
* Support long-term conversation memory

---

## 🧠 Maintainer

**Alfredo Gonzalez**
WIT Inc.
Email: [agonalez@witinc.com](mailto:agonalez@witinc.com)
