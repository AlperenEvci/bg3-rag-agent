import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def embed_and_store(input_dir, vectorstore_dir, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    os.makedirs(vectorstore_dir, exist_ok=True)
    model = SentenceTransformer(model_name)
    docs, metadatas = [], []
    for fname in os.listdir(input_dir):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(input_dir, fname), 'r', encoding='utf-8') as f:
            doc = json.load(f)
        docs.append(doc["content"])
        metadatas.append({
            "title": doc["title"],
            "url": doc["url"],
            "tags": doc["tags"],
            "chunk_id": doc["chunk_id"]
        })
    embeddings = model.encode(docs, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    faiss.write_index(index, os.path.join(vectorstore_dir, "bg3_faiss.index"))
    with open(os.path.join(vectorstore_dir, "bg3_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_dir = "data/chunked_json"
    vectorstore_dir = "embeddings/bg3_vectorstore"
    embed_and_store(input_dir, vectorstore_dir)