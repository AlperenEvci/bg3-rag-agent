from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = "embeddings/bg3_vectorstore"

def search(query, top_k=3):
    # Load model, index, and metadata
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(os.path.join(VECTORSTORE_DIR, "bg3_faiss.index"))
    with open(os.path.join(VECTORSTORE_DIR, "bg3_metadata.json"), "r", encoding="utf-8") as f:
        metadatas = json.load(f)
    # Embed the query
    embedding = model.encode([query])
    D, I = index.search(np.array(embedding).astype('float32'), top_k)
    # Print results
    for idx in I[0]:
        if idx < len(metadatas):
            print("Title:", metadatas[idx]["title"])
            print("URL:", metadatas[idx]["url"])
            print("Tags:", metadatas[idx]["tags"])
            print("Chunk ID:", metadatas[idx]["chunk_id"])
            print("---")

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    search(user_query)