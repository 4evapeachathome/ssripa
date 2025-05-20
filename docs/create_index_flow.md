# Create Index Flow Diagram

```
+------------------------+
|         Start          |
+------------------------+
           |
           v
+------------------------+
| Initialize Transformer  | ← SentenceTransformer
| (all-mpnet-base-v2)    |   'all-mpnet-base-v2'
+------------------------+
           |
           v
+------------------------+
|     Load Dataset       | ← Python built-in
| (ssripa_dataset.txt)   |   open(), read()
+------------------------+
           |
           v
+------------------------+
|   Split Documents      | ← Python str.split()
|    (by double \n)      |   list comprehension
+------------------------+
           |
           v
[Embedding Process]------+
|  > Generate Vectors    | ← SentenceTransformer
|  > Convert to NumPy    | ← NumPy array()
|  > Create FAISS Index  | ← FAISS(5z) IndexFlatL2
+------------------------+
           |
           v
+------------------------+
|   Add to FAISS Index   | ← FAISS index.add()
|     (IndexFlatL2)      |   L2 = Euclidean dist
+------------------------+
           |
           v
[Save Process]-----------+
| > Save FAISS Index     | ← FAISS write_index()
|   (ssripa_index.faiss) | ← Python file write()
| > Save Original Text   |   for metadata
|   (ssripa_metadata.txt)|
+------------------------+
           |
           v
+------------------------+
|          End           |
+------------------------+
```

Key Libraries:
- SentenceTransformer: For generating text embeddings
- NumPy: For array operations and data type conversion
- FAISS: For efficient similarity search indexing
- Python built-ins: For file I/O and text processing

