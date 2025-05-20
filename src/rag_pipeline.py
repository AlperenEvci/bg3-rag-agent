from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore import InMemoryDocstore
from langchain_core.documents import Document
from src.llm import llm
import os
import sys
import json
import faiss

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

# Initialize the embedding model - same one used to create the embeddings
print("Initializing embedding model...", file=sys.stderr)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Loading FAISS vectorstore using direct method...", file=sys.stderr)
try:
    # Method 3: Direct FAISS library loading
    print(f"Reading FAISS index directly from {index_path}", file=sys.stderr)
    faiss_index_obj = faiss.read_index(index_path)
    print(f"Successfully read FAISS index. Index has {faiss_index_obj.ntotal} vectors.", file=sys.stderr)

    print(f"Loading and processing metadata from {metadata_path}", file=sys.stderr)
    with open(metadata_path, 'r', encoding='utf-8') as f:
        raw_metadata = json.load(f)

    docstore_reconstruction = {}
    index_to_docstore_id_reconstruction = {}

    if isinstance(raw_metadata, list):
        print(f"Metadata is a list with {len(raw_metadata)} items. Reconstructing...", file=sys.stderr)
        for i, item in enumerate(raw_metadata):
            doc_id = f"doc_{i}"
            if isinstance(item, dict):
                # Assuming item is a dict where the whole item is metadata, and page_content is not in this file
                doc = Document(page_content="", metadata=item)
            else:
                print(f"Warning: Could not interpret item {i} in metadata list: {item}", file=sys.stderr)
                continue 
            
            docstore_reconstruction[doc_id] = doc
            index_to_docstore_id_reconstruction[i] = doc_id
        
        if not docstore_reconstruction:
            raise ValueError("Could not reconstruct any documents from list in metadata.")
        print(f"Reconstructed from list: {len(docstore_reconstruction)} docs.", file=sys.stderr)
    else:
        # If your metadata is structured differently (e.g., a dict with "docstore" and "index_to_docstore_id" keys)
        # you would add that logic here, similar to test_embeddings.py.
        # For now, sticking to the list-of-dicts structure observed in bg3_metadata.json
        raise ValueError(f"Unsupported metadata format in {metadata_path}. Expected a list of metadata dictionaries.")

    final_docstore = InMemoryDocstore(docstore_reconstruction)
    final_index_to_docstore_id = {int(k): v for k, v in index_to_docstore_id_reconstruction.items()}

    if faiss_index_obj.ntotal != len(final_index_to_docstore_id):
        print(f"WARNING: FAISS index has {faiss_index_obj.ntotal} vectors, "
              f"but reconstructed index_to_docstore_id has {len(final_index_to_docstore_id)} entries.", file=sys.stderr)
        # Adjusting final_index_to_docstore_id to match faiss_index_obj.ntotal
        # This ensures that every vector in the FAISS index has a corresponding entry in index_to_docstore_id,
        # and vice-versa, preventing potential errors during FAISS object construction or querying.
        
        # Option 1: If metadata is more than vectors, trim metadata mapping
        if len(final_index_to_docstore_id) > faiss_index_obj.ntotal:
            print(f"Adjusting index_to_docstore_id: Trimming to first {faiss_index_obj.ntotal} items.", file=sys.stderr)
            final_index_to_docstore_id = {
                i: final_index_to_docstore_id[i] 
                for i in range(faiss_index_obj.ntotal) 
                if i in final_index_to_docstore_id # Ensure key exists before trying to access
            }
        # Option 2: If vectors are more than metadata (less common if metadata was generated with index)
        # This case might mean some vectors won't have retrievable documents.
        # The FAISS constructor might handle this, or it might error if a vector ID is requested that's not in index_to_docstore_id.
        # For safety, ensure all referred doc_ids in final_index_to_docstore_id exist in final_docstore.
        # The current loop structure for building docstore and index_to_docstore_id ensures this for 0..N-1 mapping.

    vectorstore = FAISS(
        embedding_function=embeddings,  # Pass the HuggingFaceEmbeddings instance directly
        index=faiss_index_obj,
        docstore=final_docstore,
        index_to_docstore_id=final_index_to_docstore_id
    )
    print("FAISS vectorstore loaded successfully using direct method!", file=sys.stderr)

except Exception as e:
    print(f"Error loading vectorstore: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
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
