import os
from dotenv import load_dotenv
from google import genai
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()
client = genai.Client()

# ---- Connect to the same database ingest.py built ----
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="dsa_notes",
    embedding_function=embedding_fn
)

def retrieve_context(question, n_results=5):
    """Search the knowledge base for the most relevant chunks."""
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    chunks = results["documents"][0]

    print("\n[DEBUG] Retrieved chunks:")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---\n{chunk[:150]}...\n")

    return "\n\n---\n\n".join(chunks)

print("Luna — type 'quit' to exit\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        break

    # Step 1: search your notes for relevant chunks
    context = retrieve_context(user_input)

    # Step 2: hand those chunks to Gemini along with the question
    interaction = client.interactions.create(
        model="gemini-3.5-flash",
        system_instruction=(
            "You are Luna, a CSE tutor for a college student preparing for exams and placements. "
            "You will be given CONTEXT retrieved from the student's own notes, followed by their QUESTION. "
            "Base your answer primarily on the given context when it's relevant. If the context doesn't "
            "contain the answer, say so and answer from your own knowledge instead. "
            "STRICT RULE: Keep every answer under 100 words unless the student explicitly says "
            "'explain in detail', 'go deep', or 'give an example'. Format every answer as short bullet points "
            "(3-6 bullets), each bullet a single clear sentence — not a paragraph, not headers, not tables, "
            "no code blocks unless the student asks for code specifically. Never sacrifice correctness for "
            "brevity — just be selective about what you include."
        ),
        input=f"CONTEXT:\n{context}\n\nQUESTION:\n{user_input}",
    )

    print("Luna:", interaction.output_text)
    print()