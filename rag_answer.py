import os
from openai import OpenAI
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize sentence transformer for local embeddings
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Load FAISS index
index = faiss.read_index("data/ssripa_index.faiss")

# Load metadata (original chunks)
with open("data/ssripa_metadata.txt", "r", encoding="utf-8") as f:
    documents = [chunk.strip() for chunk in f.read().split("\n\n") if chunk.strip()]

def get_embedding(text):
    """Generate embeddings using local sentence-transformers model"""
    embedding = embedding_model.encode([text])[0]
    return embedding.astype('float32')

def retrieve_similar_chunks(query, k=1, threshold=1.0):
    """Retrieve similar chunks with similarity scores.
    
    Args:
        query (str): The query text
        k (int): Number of results to retrieve
        threshold (float): Maximum L2 distance threshold (lower = more similar)
    
    Returns:
        list: List of dictionaries containing content and similarity metrics
    """
    query_embedding = get_embedding(query).astype("float32")
    distances, indices = index.search(np.array([query_embedding]), k)
    
    results = []
    # Iterate over the distances and indices in parallel using zip
    # This combines the two lists into a list of tuples
    # e.g. [(dist_1, idx_1), (dist_2, idx_2), ...]
    for dist, idx in zip(distances[0], indices[0]):
        if dist <= threshold:
            results.append({
                "content": documents[idx],
                "similarity_score": float(1 / (1 + dist)),  # Convert to similarity (0-1)
                "distance": float(dist)
            })
    return results

def generate_answer(query):
    """Generate an answer using retrieved context with confidence scores.
    
    Args:
        query (str): The user's question
    
    Returns:
        dict: Answer and retrieval metadata
    """
    relevant_chunks = retrieve_similar_chunks(query)
    
    if not relevant_chunks:
        return {
            "answer": "I couldn't find relevant information to answer your question accurately.",
            "confidence": 0.0,
            "sources": []
        }
    
    # Sort chunks by similarity score and format context
    relevant_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
    context_parts = []
    sources = []
    
    for i, chunk in enumerate(relevant_chunks, 1):
        context_parts.append(f"[{i}] {chunk['content']}")
        sources.append({
            "chunk_id": i,
            "similarity_score": chunk['similarity_score'],
            "distance": chunk['distance']
        })
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""
You are a helpful assistant answering questions based on an action plan and severity document.
Use only the provided context to answer. Reference the source numbers [1], [2], etc. when using information.

Context:
{context}

Question:
{query}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )

    # Calculate overall confidence based on similarity scores
    avg_similarity = sum(chunk['similarity_score'] for chunk in relevant_chunks) / len(relevant_chunks)
    
    return {
        "answer": response.choices[0].message.content.strip(),
        "confidence": float(avg_similarity),
        "sources": sources
    }

def generate_consolidated_answer(queries):
    """Process multiple questions and provide a consolidated, natural response with sources.
    
    Args:
        queries (list): List of questions from the user
    
    Returns:
        dict: Consolidated answer with metadata
    """
    # Collect context and questions with metadata
    all_contexts = []
    query_results = {}
    
    for query in queries:
        chunks = retrieve_similar_chunks(query)
        if chunks:  # Only include if relevant chunks were found
            query_results[query] = chunks
            all_contexts.extend(chunks)
    
    if not all_contexts:
        return {
            "answer": "I couldn't find relevant information to answer your questions accurately.",
            "confidence": 0.0,
            "sources": [],
            "query_confidences": {}
        }
    
    # Remove duplicates while preserving order and tracking sources
    unique_contexts = []
    seen = set()
    context_parts = []
    sources = []
    
    for i, chunk in enumerate(all_contexts, 1):
        if chunk['content'] not in seen:
            seen.add(chunk['content'])
            unique_contexts.append(chunk)
            context_parts.append(f"[{i}] {chunk['content']}")
            sources.append({
                "chunk_id": i,
                "similarity_score": chunk['similarity_score'],
                "distance": chunk['distance']
            })
    
    context = "\n\n".join(context_parts)
    questions_text = "\n".join([f"- {q}" for q in queries])
    
    prompt = f"""
You are a helpful assistant analyzing multiple related questions about a situation.
Use the provided context to answer and reference source numbers [1], [2], etc. when using information.

Context from our knowledge base:
{context}

The person has shared these concerns:
{questions_text}

Please provide a natural, flowing response that:
1. Acknowledges their situation
2. Analyzes the patterns across their questions
3. Provides relevant advice and action steps
4. Maintains an empathetic and supportive tone
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=800
    )

    # Calculate confidence scores
    query_confidences = {}
    for query, chunks in query_results.items():
        avg_similarity = sum(chunk['similarity_score'] for chunk in chunks) / len(chunks)
        query_confidences[query] = float(avg_similarity)
    
    overall_confidence = sum(query_confidences.values()) / len(query_confidences)
    
    return {
        "answer": response.choices[0].message.content.strip(),
        "confidence": overall_confidence,
        "sources": sources,
        "query_confidences": query_confidences
    }

def batch_generate_answers(queries):
    """Process multiple questions in batch and return their answers."""
    answers = []
    for query in queries:
        try:
            answer = generate_answer(query)
            answers.append({"question": query, "answer": answer})
        except Exception as e:
            answers.append({"question": query, "error": str(e)})
    return answers

# Example usage
if __name__ == "__main__":
    print(f"Loaded {len(documents)} documents from metadata")
    while True:
        print("\n1. Ask a single question")
        print("2. Share multiple concerns")
        print("3. Exit")
        choice = input("\nChoose an option (1-3): ")
        
        if choice == "3":
            break
        elif choice == "1":
            user_query = input("\n❓ Please share your concern: ")
            try:
                answer = generate_answer(user_query)
                print("\n💭 Response:", answer)
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
        elif choice == "2":
            questions = []
            print("\nPlease share your concerns one by one (type 'done' when finished):")
            while True:
                question = input("\n❓ Your concern: ")
                if question.lower() == 'done':
                    break
                questions.append(question)
            
            if questions:
                print("\nAnalyzing your situation...\n")
                try:
                    consolidated_response = generate_consolidated_answer(questions)
                    print("\n💭 Here's my analysis of your situation:\n")
                    print(consolidated_response)
                except Exception as e:
                    print(f"\n❌ Error: {str(e)}")
        else:
            print("\n❌ Invalid choice. Please try again.")
