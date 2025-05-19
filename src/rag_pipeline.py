from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from src.llm import llm
import os
import sys
import json
import shutil
import pickle

# Check if the vectorstore directory exists
vectorstore_dir = "embeddings/bg3_vectorstore"
if not os.path.exists(vectorstore_dir):
    print(f"Error: Directory not found: {vectorstore_dir}", file=sys.stderr)
    print("Current directory:", os.getcwd(), file=sys.stderr)
    print("Available directories:", os.listdir(), file=sys.stderr)
    raise FileNotFoundError(f"Directory not found: {vectorstore_dir}")

# Check if the index file exists
index_path = os.path.join(vectorstore_dir, "bg3_faiss.index")
metadata_path = os.path.join(vectorstore_dir, "bg3_metadata.json")

if not os.path.exists(index_path):
    print(f"Error: Index file not found at {index_path}", file=sys.stderr)
    print("Available files in directory:", os.listdir(vectorstore_dir), file=sys.stderr)
    raise FileNotFoundError(f"Index file not found at {index_path}")

if not os.path.exists(metadata_path):
    print(f"Error: Metadata file not found at {metadata_path}", file=sys.stderr)
    print("Available files in directory:", os.listdir(vectorstore_dir), file=sys.stderr)
    raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

# Create standard filenames expected by older LangChain versions
standard_index_path = os.path.join(vectorstore_dir, "bg3_faiss.index")
standard_docstore_path = os.path.join(vectorstore_dir, "docstore.pkl")

# Copy index file with the expected name
if not os.path.exists(standard_index_path):
    print(f"Creating copy of index file at {standard_index_path}", file=sys.stderr)
    shutil.copy(index_path, standard_index_path)

# Convert JSON metadata to pickle for older versions
if not os.path.exists(standard_docstore_path):
    print(f"Converting metadata to pickle at {standard_docstore_path}", file=sys.stderr)
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(standard_docstore_path, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Error converting metadata: {e}", file=sys.stderr)

print("Loading FAISS vectorstore...", file=sys.stderr)
try:
    # Try loading with the standard approach for older LangChain versions
    vectorstore = FAISS.load_local(
        vectorstore_dir,
        "sentence-transformers/all-MiniLM-L6-v2",
        allow_dangerous_deserialization=True
    )
    print("FAISS vectorstore loaded successfully!", file=sys.stderr)
except Exception as e:
    print(f"Error loading vectorstore: {e}", file=sys.stderr)
    raise

# Define prompt template
prompt_template = """
You are an expert assistant for Baldur's Gate 3. Use the following context to answer the user's question.

Context:
{context}

Question: 
{question}

Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Create RAG pipeline
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)
