from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# Initialize the sentence transformer model
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Read the dataset
with open("data/ssripa_dataset.txt", "r", encoding="utf-8") as f:
    documents = [chunk.strip() for chunk in f.read().split("\n\n") if chunk.strip()]

# Create embeddings
embeddings = []
for doc in documents:
    embedding = embedding_model.encode([doc])[0]
    embeddings.append(embedding)

embeddings = np.array(embeddings).astype('float32')

# Create FAISS index
dimension = embeddings.shape[1]  # Get the dimension of embeddings
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save the index
faiss.write_index(index, "data/ssripa_index.faiss")

# Save the metadata
with open("data/ssripa_metadata.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(documents))

print("Index and metadata created successfully!")
