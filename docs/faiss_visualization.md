# FAISS Index Data Structure and Similarity Scoring

```
Original Text Documents
+-------------------+
| Document 1        |     +------------------+
| "Hello world..."  | --> | Sentence         |
+-------------------+     | Transformer      | 
                         | (all-mpnet-base-v2)|
+-------------------+    +------------------+
| Document 2        |            |
| "RAG system..."   | --------→  |  
+-------------------+            |
                                v
Vector Space (768-dimensional for all-mpnet-base-v2)
+------------------------------------------------+
|                                                 |
| Doc1 Vector: [0.1, 0.3, -0.2, ..., 0.4]        |
|                     ↓                           |
| Doc2 Vector: [0.2, -0.1, 0.5, ..., -0.3]       |
|                     ↓                           |
| Doc3 Vector: [-0.3, 0.4, 0.1, ..., 0.2]        |
|                                                 |
+------------------------------------------------+
                     |
                     v
FAISS IndexFlatL2 Structure
+------------------------------------------------+
| Internal Vector Store                           |
| +--------------------------------------------+ |
| | Vector Data (float32)                       | |
| | • Contiguous memory layout                  | |
| | • Fixed dimension (768)                     | |
| | • Optimized for L2 distance computation     | |
| +--------------------------------------------+ |
|                                                |
| Search Index                                   |
| +--------------------------------------------+ |
| | • No compression (Flat index)               | |
| | • Exact nearest neighbor search (k=1)       | |
| | • L2 distance threshold = 1.5               | |
| +--------------------------------------------+ |
+------------------------------------------------+
                     |
                     v
Similarity Scoring
+------------------------------------------------+
| Query Result Processing                         |
| +--------------------------------------------+ |
| | 1. Get L2 Distance (d)                      | |
| | 2. Filter: d <= 1.5                        | |
| | 3. Convert to Similarity: s = 1/(1+d)      | |
| | 4. Package with Metadata:                   | |
| |    {                                       | |
| |      "content": original_text,             | |
| |      "similarity_score": s,               | |
| |      "distance": d                        | |
| |    }                                       | |
| +--------------------------------------------+ |
+------------------------------------------------+
                     |
                     v
On-Disk Storage
+------------------------+    +----------------------+
| ssripa_index.faiss     |    | ssripa_metadata.txt |
| (Binary FAISS format)  |    | (Original texts)    |
+------------------------+    +----------------------+

Query Process Example:
1. Input Query: "How does RAG work?"
2. Generate Vector: [0.2, -0.1, 0.4, ..., 0.1]
3. Find Nearest Match:
   • Calculate L2 distances
   • Get closest vector (k=1)
   • Check threshold (≤ 1.5)
4. Score and Return:
   • Similarity = 1/(1 + L2_distance)
   • Include metadata and sources
```

## Key Components:

1. **Vector Generation**:
   - Dimension: 768 (all-mpnet-base-v2)
   - Type: float32
   - Normalized embeddings

2. **Search Parameters**:
   - k=1 (single best match)
   - L2 distance threshold: 1.5
   - Exact nearest neighbor search

3. **Similarity Metrics**:
   - L2 Distance: Raw Euclidean distance
   - Similarity Score: 1/(1 + distance)
   - Range: 0-1 (higher = more similar)

4. **Result Structure**:
   - Original content
   - Similarity score
   - L2 distance
   - Source references

5. **Quality Control**:
   - Distance threshold filtering
   - Confidence scoring
   - Source attribution
