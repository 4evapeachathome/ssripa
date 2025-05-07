import os

CHUNK_SIZE = 300  # characters, you can change this later

def load_documents(data_dir="data"):
    docs = []
    for filename in os.listdir(data_dir):
        path = os.path.join(data_dir, filename)
        if os.path.isfile(path) and path.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as file:
                docs.append(file.read())
    return docs

def chunk_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def main():
    raw_docs = load_documents()
    print(f"Loaded {len(raw_docs)} document(s).")

    all_chunks = []
    for doc in raw_docs:
        chunks = chunk_text(doc)
        all_chunks.extend(chunks)
    
    print(f"Generated {len(all_chunks)} text chunks.")
    for i, chunk in enumerate(all_chunks[:3]):  # Show a sample
        print(f"\n--- Chunk {i+1} ---\n{chunk}\n")

if __name__ == "__main__":
    main()
