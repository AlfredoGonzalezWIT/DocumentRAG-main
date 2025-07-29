# main.py

import os
import fitz  # PyMuPDF
from embedding_helper import EmbeddingHelper
from faiss_indexing_helper import FAISSIndexingHelper  # renamed helper for FAISS
from dotenv import load_dotenv
import logging

# Configure logging and make sure log folder exists
log_folder = "RAG"
os.makedirs(log_folder, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_folder, 'main.log'),
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = " ".join(page.get_text() for page in doc)
        doc.close()
        logging.info(f"Extracted text from '{pdf_path}'")
        return text
    except Exception as e:
        logging.error(f"Failed to extract text from '{pdf_path}': {e}")
        raise

def main():
    # Load environment variables
    load_dotenv(dotenv_path="local.env")

    # Initialize embedding and indexing helpers
    embedder = EmbeddingHelper(env_file="local.env")
    indexing_helper = FAISSIndexingHelper()

    # Folder containing PDFs
    pdf_folder = os.path.join("RAG", "pdfs")
    if not os.path.exists(pdf_folder):
        logging.error(f"PDF folder '{pdf_folder}' does not exist.")
        print(f"PDF folder '{pdf_folder}' does not exist.")
        return

    # Loop through all PDFs
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            base_doc_id = os.path.splitext(filename)[0]
            metadata = {"source": filename}

            try:
                # Step 1: Extract text
                text_summary = extract_text_from_pdf(pdf_path)
                print(f"\n📄 Processing: {filename}")
                print("Extracted Text Preview:")
                print(text_summary[:300], "...\n")

                # Step 2: Split text into 3000-character chunks
                chunk_size = 3000
                chunks = [text_summary[i:i + chunk_size] for i in range(0, len(text_summary), chunk_size)]

                # Step 3: Generate embedding for each chunk and index it
                for j, chunk in enumerate(chunks):
                    try:
                        embedding_vector = embedder.get_embedding(chunk)
                        chunk_id = f"{base_doc_id}_part{j+1}"
                        indexing_helper.add_document(
                            doc_id=chunk_id,
                            content=chunk,
                            embedding=embedding_vector,
                            metadata=metadata
                        )
                        print(f"✅ Indexed chunk {j+1} of {filename}")
                    except Exception as chunk_error:
                        logging.error(f"Error embedding chunk {j+1} of '{filename}': {chunk_error}")
                        print(f"❌ Failed to index chunk {j+1} — {chunk_error}")

            except Exception as e:
                logging.error(f"Error processing '{filename}': {e}")
                print(f"❌ Failed: {filename} — {e}")

    # Save FAISS index and metadata
    indexing_helper.save()
    print("💾 FAISS index and metadata saved.")

if __name__ == "__main__":
    main()
