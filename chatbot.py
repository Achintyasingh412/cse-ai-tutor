import os
from dotenv import load_dotenv
from google import genai
import chromadb
from chromadb.utils import embedding_functions
from memory_db import get_recent_conversation_history, log_conversation

load_dotenv()
client = genai.Client()

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="dsa_notes",
    embedding_function=embedding_fn
)

def retrieve_context(question, n_results=5):
    results = collection.query(query_texts=[question], n_results=n_results)
    return "\n\n---\n\n".join(results["documents"][0])

print("Luna — type 'quit' to exit\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        break
    chat_history = get_recent_conversation_history()
    history_text = "\n".join(
        [f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history]
    )

    context = retrieve_context(user_input)

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
            "brevity — just be selective about what you include. "
            "IMPORTANT FORMATTING RULE: Never use LaTeX or math notation like \\(O(n)\\) or $O(n)$. "
            "Always write complexity and math in plain text instead, like O(n log n) or O(n squared) — "
            "no dollar signs, no backslashes, no special math formatting of any kind."
        ),
        input=f"CONTEXT:\n{context}\n\nQUESTION:\n{user_input}",
    )

    print("Luna:", interaction.output_text)
    print()