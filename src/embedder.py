import os
import json

def chunk_text(text,chunk_size=500, overlap=50):
    """
    Splits the text into chunks of specified size with overlap.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks

def chunk_json_files(input_dir, output_dir, chunk_size=500, overlap=50):
    """Reads JSON files, chunks their content, and saves chunked docs."""
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(input_dir, fname), 'r', encoding='utf-8') as f:
            doc = json.load(f)
        chunks = chunk_text(doc['content'], chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            chunked_doc = {
                "title": doc["title"],
                "url": doc["url"],
                "tags": doc["tags"],
                "content": chunk,
                "chunk_id": f"{fname.replace('.json','')}_chunk_{i}"
            }
            out_path = os.path.join(output_dir, f"{fname.replace('.json','')}_chunk_{i}.json")
            with open(out_path, 'w', encoding='utf-8') as out_f:
                json.dump(chunked_doc, out_f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    input_dir = "data/parsed_json"
    output_dir = "data/chunked_json"
    chunk_json_files(input_dir, output_dir)  