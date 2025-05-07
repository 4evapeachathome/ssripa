import os
from openai import OpenAI
import faiss
import numpy as np

# Setup your OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load the dataset
with open("./data/ssripa_dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split the text into chunks based on double line breaks
chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

# Function to create embeddings
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Generate and store embeddings
dimension = 1536  # dimension for text-embedding-ada-002
index = faiss.IndexFlatL2(dimension)
metadata = []

for i, chunk in enumerate(chunks):
    embedding = get_embedding(chunk)
    index.add(np.array([embedding]).astype("float32"))
    metadata.append(chunk)

# Save the index and metadata
faiss.write_index(index, "ssripa_index.faiss")
with open("ssripa_metadata.txt", "w", encoding="utf-8") as f:
    for entry in metadata:
        f.write(entry + "\n\n")

print("✅ Embeddings generated and stored locally.")
