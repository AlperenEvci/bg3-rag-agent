"""
Test script to debug loading the bg3 vector store embeddings.
"""
print("Starting test script...")
import os
import sys
import json
import pickle
import shutil
import faiss # Assuming faiss-cpu is installed

# Try to import from langchain_community first, as per your installed packages
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.docstore import InMemoryDocstore
except ImportError:
    print("Could not import from langchain_community. Trying core langchain...")
    # Fallback for older structures if community is not resolving for the tool
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.docstore import InMemoryDocstore # Or from langchain_community.docstore

from langchain_core.documents import Document


# Set up paths
vectorstore_dir = os.path.abspath("embeddings/bg3_vectorstore")
index_path = os.path.join(vectorstore_dir, "bg3_faiss.index")
metadata_path = os.path.join(vectorstore_dir, "bg3_metadata.json")

print(f"Working directory: {os.getcwd()}")
print(f"Absolute vectorstore_dir: {vectorstore_dir}")
print(f"Checking paths:")
print(f"Vector store directory exists: {os.path.exists(vectorstore_dir)}")
print(f"Index file ({index_path}) exists: {os.path.exists(index_path)}")
print(f"Metadata file ({metadata_path}) exists: {os.path.exists(metadata_path)}")

if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
    print("ERROR: Essential embedding files (bg3_faiss.index or bg3_metadata.json) not found. Exiting.")
    sys.exit(1)

# Initialize embedding model
print("Initializing embedding model...")
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    print(f"ERROR initializing HuggingFaceEmbeddings: {e}")
    sys.exit(1)

# --- Helper function to create PKL ---
def create_pkl_from_json_metadata(json_path, pkl_path):
    print(f"Attempting to create {pkl_path} from {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_metadata = json.load(f)
    
    docstore_reconstruction = {}
    index_to_docstore_id_reconstruction = {}

    if isinstance(raw_metadata, dict) and "docstore" in raw_metadata and "index_to_docstore_id" in raw_metadata:
        # This is the ideal pre-structured format
        docstore_data = raw_metadata["docstore"]
        for doc_id, doc_content_meta in docstore_data.items():
            if isinstance(doc_content_meta, dict): # Expected: {'page_content': '...', 'metadata': {...}}
                 docstore_reconstruction[doc_id] = Document(page_content=doc_content_meta.get("page_content",""), metadata=doc_content_meta.get("metadata",{}))
            else: # Legacy? If doc_content_meta is just the page_content string
                 docstore_reconstruction[doc_id] = Document(page_content=str(doc_content_meta), metadata={})

        index_to_docstore_id_reconstruction = raw_metadata["index_to_docstore_id"]
        print(f"Reconstructed from structured JSON: {len(docstore_reconstruction)} docs.")
    elif isinstance(raw_metadata, list):
        # Common case: list of [page_content, metadata_dict] or list of Document-like dicts
        print(f"Metadata is a list with {len(raw_metadata)} items. Attempting reconstruction...")
        for i, item in enumerate(raw_metadata):
            doc_id = f"doc_{i}"
            if isinstance(item, dict) and "page_content" in item:
                doc = Document(page_content=item["page_content"], metadata=item.get("metadata", {}))
            elif isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], dict):
                doc = Document(page_content=item[0], metadata=item[1])
            else:
                print(f"Warning: Could not interpret item {i} in metadata list: {item}")
                continue
            docstore_reconstruction[doc_id] = doc
            index_to_docstore_id_reconstruction[i] = doc_id
        if not docstore_reconstruction:
            raise ValueError("Could not reconstruct any documents from the list in metadata JSON.")
        print(f"Reconstructed from list: {len(docstore_reconstruction)} docs.")
    else:
        raise ValueError(f"Unsupported format in {json_path}. Expected dict with 'docstore'/'index_to_docstore_id' or a list.")

    docstore = InMemoryDocstore(docstore_reconstruction)
    with open(pkl_path, 'wb') as pf:
        pickle.dump((docstore, index_to_docstore_id_reconstruction), pf)
    print(f"Successfully created {pkl_path}")
    return True

# --- Method 1: FAISS.load_local (expecting default names: index.faiss, index.pkl) ---
print("\n--- Attempting Method 1: FAISS.load_local (default names) ---")
default_faiss_path = os.path.join(vectorstore_dir, "index.faiss")
default_pkl_path = os.path.join(vectorstore_dir, "index.pkl")
method1_success = False

try:
    # Ensure index.faiss exists
    if not os.path.exists(default_faiss_path) and os.path.exists(index_path):
        shutil.copy(index_path, default_faiss_path)
        print(f"Copied {index_path} -> {default_faiss_path} for Method 1")
    
    # Ensure index.pkl exists
    if not os.path.exists(default_pkl_path) and os.path.exists(metadata_path):
        create_pkl_from_json_metadata(metadata_path, default_pkl_path)

    if os.path.exists(default_faiss_path) and os.path.exists(default_pkl_path):
        vectorstore = FAISS.load_local(
            folder_path=vectorstore_dir, # Looks for index.faiss and index.pkl here
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        print("SUCCESS (Method 1): FAISS vectorstore loaded using default names!")
        method1_success = True
    else:
        print("Method 1: Default files (index.faiss/index.pkl) not found or created. Skipping load attempt.")

except Exception as e:
    print(f"ERROR (Method 1): {str(e)}")
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(f"Detail: {exc_type}, {fname}, {exc_tb.tb_lineno}")

# --- Method 2: FAISS.load_local with explicit original index_name (e.g., "bg3_faiss") ---
# This expects "bg3_faiss.index" and "bg3_faiss.pkl"
print("\n--- Attempting Method 2: FAISS.load_local (explicit index_name='bg3_faiss') ---")
custom_pkl_name = "bg3_faiss.pkl" # To match index_name="bg3_faiss"
custom_pkl_path = os.path.join(vectorstore_dir, custom_pkl_name)
method2_success = False

try:
    # Ensure bg3_faiss.pkl exists
    if not os.path.exists(custom_pkl_path) and os.path.exists(metadata_path):
         create_pkl_from_json_metadata(metadata_path, custom_pkl_path)

    if os.path.exists(index_path) and os.path.exists(custom_pkl_path): # original .index, custom .pkl
        vectorstore = FAISS.load_local(
            folder_path=vectorstore_dir,
            embeddings=embeddings,
            index_name="bg3_faiss", # Expects bg3_faiss.index and bg3_faiss.pkl
            allow_dangerous_deserialization=True
        )
        print("SUCCESS (Method 2): FAISS vectorstore loaded using index_name='bg3_faiss'!")
        method2_success = True
    else:
        print(f"Method 2: Original index ({index_path}) or its pkl ({custom_pkl_path}) not found/created. Skipping.")

except Exception as e2:
    print(f"ERROR (Method 2): {str(e2)}")
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(f"Detail: {exc_type}, {fname}, {exc_tb.tb_lineno}")


# --- Method 3: Direct FAISS library loading (using original bg3_faiss.index and bg3_metadata.json) ---
print("\n--- Attempting Method 3: Direct FAISS library loading ---")
method3_success = False
try:
    print(f"Method 3: Reading FAISS index directly from {index_path}")
    faiss_index_obj = faiss.read_index(index_path)
    print(f"Successfully read FAISS index. Index has {faiss_index_obj.ntotal} vectors.")

    print(f"Method 3: Loading and processing metadata from {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        raw_metadata = json.load(f)

    docstore_reconstruction = {}
    index_to_docstore_id_reconstruction = {}

    if isinstance(raw_metadata, dict) and "docstore" in raw_metadata and "index_to_docstore_id" in raw_metadata:
        docstore_data = raw_metadata["docstore"]
        for doc_id, doc_content_meta in docstore_data.items():
            if isinstance(doc_content_meta, dict):
                 docstore_reconstruction[doc_id] = Document(page_content=doc_content_meta.get("page_content",""), metadata=doc_content_meta.get("metadata",{}))
            else:
                 docstore_reconstruction[doc_id] = Document(page_content=str(doc_content_meta), metadata={})
        index_to_docstore_id_reconstruction = raw_metadata["index_to_docstore_id"]
        print(f"Method 3: Reconstructed from structured JSON: {len(docstore_reconstruction)} docs.")
    elif isinstance(raw_metadata, list):
        print(f"Method 3: Metadata is a list with {len(raw_metadata)} items. Reconstructing...")
        for i, item in enumerate(raw_metadata):
            doc_id = f"doc_{i}"
            if isinstance(item, dict):
                if "page_content" in item:
                    # Case 1: item is a dict with 'page_content' (and optional 'metadata' key)
                    doc = Document(page_content=item["page_content"], metadata=item.get("metadata", {}))
                else:
                    # Case 2: item is a dict without 'page_content'. Treat the whole item as metadata.
                    # This handles the structure of bg3_metadata.json
                    doc = Document(page_content="", metadata=item)
            elif isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], dict):
                # Case 3: item is a tuple (page_content, metadata_dict)
                doc = Document(page_content=item[0], metadata=item[1])
            else:
                print(f"Warning (Method 3): Could not interpret item {i}: {item}")
                continue # Skip if format is unexpected
            
            docstore_reconstruction[doc_id] = doc
            index_to_docstore_id_reconstruction[i] = doc_id
        if not docstore_reconstruction:
            raise ValueError("Method 3: Could not reconstruct any documents from list in metadata.")
        print(f"Method 3: Reconstructed from list: {len(docstore_reconstruction)} docs.")
    else:
        raise ValueError(f"Method 3: Unsupported format in {metadata_path}.")

    final_docstore = InMemoryDocstore(docstore_reconstruction)
    
    # Ensure index_to_docstore_id keys are integers if FAISS index expects that
    final_index_to_docstore_id = {int(k): v for k, v in index_to_docstore_id_reconstruction.items()}


    if faiss_index_obj.ntotal != len(final_index_to_docstore_id):
        print(f"WARNING (Method 3): FAISS index has {faiss_index_obj.ntotal} vectors, but reconstructed index_to_docstore_id has {len(final_index_to_docstore_id)} entries.")
        # Potentially trim or pad index_to_docstore_id if necessary, or error out
        # For now, we'll proceed, but this is a common source of issues.
        # If len(final_index_to_docstore_id) > faiss_index_obj.ntotal, FAISS might complain.
        # If len(final_index_to_docstore_id) < faiss_index_obj.ntotal, some vectors won't have metadata.
        # Let's try to ensure they match, assuming index_to_docstore_id should map 0 to ntotal-1
        if len(final_index_to_docstore_id) > faiss_index_obj.ntotal:
            print("Adjusting index_to_docstore_id to match FAISS index ntotal by taking the first ntotal items.")
            final_index_to_docstore_id = {i: final_index_to_docstore_id[i] for i in range(faiss_index_obj.ntotal) if i in final_index_to_docstore_id}
        elif len(final_index_to_docstore_id) < faiss_index_obj.ntotal:
             print("index_to_docstore_id is smaller than FAISS index. Some vectors will lack metadata.")


    vectorstore = FAISS(
        embedding_function=embeddings.embed_query,
        index=faiss_index_obj,
        docstore=final_docstore,
        index_to_docstore_id=final_index_to_docstore_id
    )
    print("SUCCESS (Method 3): FAISS vectorstore loaded with direct FAISS library creation!")
    method3_success = True

except Exception as e3:
    print(f"ERROR (Method 3): {str(e3)}")
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(f"Detail: {exc_type}, {fname}, {exc_tb.tb_lineno}")

# --- Query Test if any method succeeded ---
if method1_success or method2_success or method3_success:
    print("\n--- Performing Test Query ---")
    try:
        query = "What is the Paladin class like?"
        docs = vectorstore.similarity_search(query, k=1) # vectorstore will be from the last successful method
        print(f"Test query: '{query}'")
        if docs:
            print(f"Result (page_content): {docs[0].page_content[:200]}...")
            print(f"Result (metadata): {docs[0].metadata}")
        else:
            print("Result: No documents found for the query.")
    except Exception as qe:
        print(f"ERROR during test query: {str(qe)}")
else:
    print("\nNo loading method succeeded. Cannot perform test query.")

# --- Cleanup temporary files ---
print("\n--- Cleaning up temporary files ---")
files_to_remove = [default_faiss_path, default_pkl_path, custom_pkl_path]
for f_path in files_to_remove:
    if f_path and os.path.exists(f_path): # Check if f_path is not None
        # Only remove if it's not one of the original files we want to keep
        if os.path.basename(f_path) not in ["bg3_faiss.index", "bg3_metadata.json"]:
            try:
                os.remove(f_path)
                print(f"Removed {f_path}")
            except OSError as e:
                print(f"Error removing {f_path}: {e.strerror}")
        else:
            print(f"Skipped removing original file: {f_path}")


print("\nTest script finished.")
