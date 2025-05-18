from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = "embeddings/bg3_vectorstore"

app = FastAPI()

# Load model, index, and metadata at startup
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(os.path.join(VECTORSTORE_DIR, "bg3_faiss.index"))
with open(os.path.join(VECTORSTORE_DIR, "bg3_metadata.json"), "r", encoding="utf-8") as f:
    metadatas = json.load(f)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/search")
def search(request: QueryRequest):
    embedding = model.encode([request.query])
    D, I = index.search(np.array(embedding).astype('float32'), request.top_k)
    results = []
    for idx in I[0]:
        if idx < len(metadatas):
            result = metadatas[idx].copy()
            result["score"] = float(D[0][list(I[0]).index(idx)])
            results.append(result)
    return {"results": results}