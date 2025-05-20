from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
import uuid
from fastapi.middleware.cors import CORSMiddleware
from src.rag_pipeline import qa_chain
from src.db import init_db, add_conversation, get_conversation_history

# Initialize the database on startup
init_db()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = "embeddings/bg3_vectorstore"

app = FastAPI()

# Add custom exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Extract the original request body for debugging
    body = await request.body()
    body_str = body.decode("utf-8") if body else "No body"
    print(f"Validation error: {exc}")
    print(f"Request body: {body_str}")
    print(f"Errors: {exc.errors()}")
    
    # Return detailed error information for debugging
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": body_str
        },
    )

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
    session_id: Optional[str] = None  # Optional session ID for tracking conversations

class ConversationHistoryRequest(BaseModel):
    session_id: Optional[str] = None
    limit: int = 10
    offset: int = 0

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
    """Endpoint to handle queries using the RAG pipeline and store in database"""
    print(f"Request data: {request}")  # Debug logging
    query_text = request.query
    response = qa_chain.invoke({"query": query_text})
    answer = response["result"]
    
    # Store conversation in database
    session_id = request.session_id or str(uuid.uuid4())  # Generate a new session ID if not provided
    add_conversation(
        query=query_text,
        response=answer,
        session_id=session_id
    )
    
    return {"answer": answer, "session_id": session_id}

@app.post("/history")
def get_history(request: ConversationHistoryRequest):
    """Endpoint to retrieve conversation history"""
    conversations = get_conversation_history(
        limit=request.limit,
        offset=request.offset,
        session_id=request.session_id
    )
    return {"history": conversations}

@app.get("/history/latest")
def get_latest_conversation():
    """Endpoint to get the most recent conversation"""
    conversations = get_conversation_history(limit=1)
    return {"conversation": conversations[0] if conversations else None}