from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
from fastapi.middleware.cors import CORSMiddleware
from src.rag_pipeline import qa_chain

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = "embeddings/bg3_vectorstore"

app = FastAPI()

# Add CORS middleware to allow frontend to connect to API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            
            # Find the content file and retrieve the actual content
            chunk_id = result["chunk_id"]
            content_file = os.path.join("data/chunked_json", f"{chunk_id}.json")
            try:
                if os.path.exists(content_file):
                    with open(content_file, "r", encoding="utf-8") as f:
                        chunk_data = json.load(f)
                        result["content"] = chunk_data.get("content", "")
            except Exception as e:
                print(f"Error loading content for {chunk_id}: {e}")
                
            results.append(result)
    return {"results": results}

class QueryResponse(BaseModel):
    query: str

@app.post("/query")
def query(request: QueryRequest):
    """Endpoint to handle queries using the RAG pipeline"""
    response = qa_chain.invoke({"query": request.query})
    return {"answer": response["result"]}