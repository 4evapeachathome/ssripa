# Query Processing Flow Diagram

```
User Question: "Are you feeling threatened by your partner or spouse?"

+--------------------------------+
|        User Question           | ← FastAPI
|   "Are you feeling..."         |   @app.post("/api/rag_query")
+--------------------------------+
              |
              v
+--------------------------------+
|     Generate Query Vector      | ← SentenceTransformer
|    (all-mpnet-base-v2)        |   embedding_model.encode([query])[0]
+--------------------------------+
              |
              v
+--------------------------------+
|    Load FAISS Index           | ← FAISS
|   (ssripa_index.faiss)        |   faiss.read_index("data/ssripa_index.faiss")
+--------------------------------+
              |
              v
+--------------------------------+
|    Similarity Search (k=1)     | ← FAISS + NumPy
|    1. Calculate L2 Distance    |   index.search(np.array([query_embedding]), k)
|    2. Find Nearest Neighbors   |   Returns: distances, indices
|    3. Filter by threshold      |   threshold=1.5 (L2 distance)
+--------------------------------+
              |
              v
+--------------------------------+
|    Process Results             | ← Python Dict
| - Content: Original text       |   {"content": doc,
| - Similarity: 1/(1+distance)  |    "similarity_score": score,
| - Distance: L2 distance       |    "distance": dist}
+--------------------------------+
              |
              v
+--------------------------------+
|     Generate Response          | ← OpenAI GPT-4
| 1. Source-referenced answer    |   temperature=0.3
| 2. Confidence scoring         |   max_tokens=500
| 3. Source metadata            |   [Source citations]
+--------------------------------+

Code Implementation:
```python
# 1. Vector Generation with Similarity Scoring
def retrieve_similar_chunks(query, k=1, threshold=1.5):
    query_embedding = get_embedding(query)
    distances, indices = index.search(
        np.array([query_embedding]), k
    )
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if dist <= threshold:
            results.append({
                "content": documents[idx],
                "similarity_score": 1/(1 + dist),
                "distance": dist
            })
    return results

# 2. Answer Generation with Metadata
def generate_answer(query):
    chunks = retrieve_similar_chunks(query)
    if not chunks:
        return {
            "answer": "No relevant information found.",
            "confidence": 0.0,
            "sources": []
        }
    
    # Format context with source references
    context_parts = [f"[{i}] {chunk['content']}" 
                    for i, chunk in enumerate(chunks, 1)]
    context = "\n\n".join(context_parts)
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", 
                  "content": f"Context:\n{context}\n\nQuestion:\n{query}"}],
        temperature=0.3,
        max_tokens=500
    )
    
    return {
        "answer": response.choices[0].message.content,
        "confidence": sum(c['similarity_score'] for c in chunks)/len(chunks),
        "sources": chunks
    }
```

Key Features:
------------
✓ Quality Control: L2 distance threshold filtering
✓ Confidence Scoring: Similarity metrics (0-1 scale)
✓ Source Attribution: Referenced context chunks
✓ Precise Search: Single best match (k=1)
✓ Rich Metadata: Distance and similarity metrics

Libraries Used:
--------------
• FastAPI: API endpoints
• SentenceTransformer: Vector generation
• FAISS-CPU: Similarity search
• NumPy: Array operations
• OpenAI: GPT-4 integration

Metadata Structure:
-----------------
• similarity_score: 0-1 (higher = more relevant)
• distance: L2 distance (lower = more similar)
• sources: List of referenced chunks with scores
• confidence: Overall answer confidence
