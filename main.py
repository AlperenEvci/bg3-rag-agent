"""
BG3 RAG Agent - Main Entry Point
"""
import os
import sys
import argparse
import uvicorn

def main():
    parser = argparse.ArgumentParser(description="BG3 RAG Agent CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Add chunk command
    chunk_parser = subparsers.add_parser("chunk", help="Chunk JSON files")
    
    # Add embed command
    embed_parser = subparsers.add_parser("embed", help="Create embeddings and FAISS index")
    
    # Add serve command
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    
    args = parser.parse_args()
    
    if args.command == "chunk":
        from src.embedder import chunk_json_files
        print("Chunking JSON files...")
        chunk_json_files("data/parsed_json", "data/chunked_json")
        print("Chunking complete.")
        
    elif args.command == "embed":
        from src.vectorizer import embed_and_store
        print("Creating embeddings and FAISS index...")
        embed_and_store("data/chunked_json", "embeddings/bg3_vectorstore")
        print("Embedding complete.")
        
    elif args.command == "serve":
        print(f"Starting API server at http://{args.host}:{args.port}...")
        uvicorn.run("src.api:app", host=args.host, port=args.port, reload=True)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()