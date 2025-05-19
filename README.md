# bg3-rag-agent

A Retrieval-Augmented Generation (RAG) system and API for Baldur's Gate 3, enabling semantic search and intelligent Q&A over wiki-based game knowledge.

## Features

- **Wiki Data Ingestion:** Scrape and parse Baldur's Gate 3 wiki content into structured JSON.
- **Chunking:** Split large documents into overlapping, manageable text chunks for better retrieval.
- **Embeddings:** Generate semantic embeddings for each chunk using the free `all-MiniLM-L6-v2` model.
- **Vector Search:** Store and search embeddings efficiently with FAISS.
- **API:** Query the knowledge base via a FastAPI endpoint for integration with tools or UIs.
- **Dockerized:** Fully containerized for easy setup and reproducibility.

## Project Structure

```
bg3-rag-agent/
├── data/
│   ├── raw_html/         # (Optional) Scraped HTML pages
│   ├── parsed_json/      # Parsed wiki data (title, url, content, tags)
│   └── chunked_json/     # Pre-chunked documents
├── embeddings/
│   └── bg3_vectorstore/  # FAISS index and metadata
├── src/
│   ├── scraper.py        # HTML scraper
│   ├── parser.py         # HTML to JSON converter
│   ├── embedder.py       # Chunking logic
│   ├── vectorizer.py     # Embedding + FAISS logic
│   ├── api.py            # FastAPI app
│   └── tests/            # Test scripts
├── main.py               # Entrypoint (optional)
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker build file
├── docker-compose.yml    # Docker Compose config
└── .env                  # Environment variables
```

## Quickstart

### 1. Build and Start the Environment

```pwsh
# Build the Docker image
docker-compose build

# Run chunking (if not already done)
docker-compose run --rm rag-agent python src/embedder.py

# Run embedding and FAISS index creation
docker-compose run --rm rag-agent python src/vectorizer.py

# Start the FastAPI server
docker-compose up
```

### 2. Query the API

- Visit [http://localhost:8000/docs](http://localhost:8000/docs) for interactive docs.
- Use the `/search` endpoint with a JSON body like:

```json
{
  "query": "What is the best rogue build in Baldur's Gate 3?",
  "top_k": 3
}
```

### 3. Example Query via CLI

```pwsh
docker-compose run --rm rag-agent python src/tests/query_test.py
```

## Running the Frontend

### Option 1: Simple Method

Use the included helper script to start both API and frontend:

```pwsh
# Start both API server and frontend
python start_app.py
```

This will:

1. Start the FastAPI server on port 8000
2. Start a simple HTTP server for the frontend on port 3000
3. Automatically open your browser to the frontend

### Option 2: Manual Setup

If you prefer to run them separately:

1. Start the API server:

```pwsh
# Start the API using main.py CLI
python main.py serve
# OR directly with uvicorn
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

2. Serve the frontend:

```pwsh
cd frontend
python -m http.server 3000
```

3. Open your browser to: http://localhost:3000

## Requirements

- Docker & Docker Compose (recommended)
- Or: Python 3.8+, see `requirements.txt`

## Technologies Used

- Python, FastAPI, FAISS, Sentence Transformers, LangChain, Docker

## Customization & Extending

- Add new data by placing parsed JSON in `data/parsed_json/` and rerunning chunking/embedding steps.
- Extend the API in `src/api.py` for more advanced RAG or answer generation.

## License

MIT
