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

def retrieve_similar_chunks(query, k=3):
    query_embedding = get_embedding(query).astype("float32")
    distances, indices = index.search(np.array([query_embedding]), k)
    return [documents[i] for i in indices[0]]

def generate_answer(query):
    relevant_chunks = retrieve_similar_chunks(query)
    context = "\n\n".join(relevant_chunks)

    prompt = f"""
You are a helpful assistant answering questions based on an action plan and severity document. Use only the provided context to answer.

Context:
{context}

Question:
{query}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()

def generate_consolidated_answer(queries):
    """Process multiple questions and provide a consolidated, natural response."""
    # Collect context and questions
    all_contexts = []
    for query in queries:
        relevant_chunks = retrieve_similar_chunks(query)
        all_contexts.extend(relevant_chunks)
    
    # Remove duplicates while preserving order
    unique_contexts = []
    seen = set()
    for context in all_contexts:
        if context not in seen:
            unique_contexts.append(context)
            seen.add(context)
    
    context = "\n\n".join(unique_contexts)
    print("\n🔍 Context used for answering:")
    print(context)
    questions_text = "\n".join([f"- {q}" for q in queries])

    prompt = f"""
You are an empathetic counselor analyzing multiple questions about someone's relationship situation. 
Provide a thoughtful, consolidated response that addresses all their concerns in a natural, conversational way. 
Make it feel like a real-time analysis of their situation.

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
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=800
    )

    return response.choices[0].message.content.strip()

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
