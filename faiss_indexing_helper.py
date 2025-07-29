# faiss_indexing_helper.py

import os
import faiss
import pickle
import numpy as np

class FAISSIndexingHelper:
    def __init__(self, dimension=3072, index_file="vector.index", metadata_file="documents.pkl"):
        self.dimension = dimension
        self.index_file = index_file
        self.metadata_file = metadata_file

        self.index = faiss.read_index(index_file) if os.path.exists(index_file) else faiss.IndexFlatL2(dimension)
        self.documents = self._load_metadata()

        # Match vector and document count
        if self.index.ntotal != len(self.documents):
            print(f"⚠️ Index/document mismatch: {self.index.ntotal} vs {len(self.documents)}")

    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "rb") as f:
                return pickle.load(f)
        return []

    def add_document(self, doc_id, content, embedding, metadata=None):
        self.index.add(np.array([embedding], dtype="float32"))
        self.documents.append({
            "id": doc_id,
            "content": content,
            "metadata": metadata,
        })

    def save(self):
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, "wb") as f:
            pickle.dump(self.documents, f)

    def search(self, query_vector, k=3):
        if self.index.ntotal == 0:
            print("⚠️ No vectors to search.")
            return []

        D, I = self.index.search(np.array([query_vector], dtype="float32"), k)
        return [self.documents[i] for i in I[0] if 0 <= i < len(self.documents)]
