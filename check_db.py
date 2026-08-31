import chromadb
from chromadb.utils import embedding_functions
from collections import Counter
from memory_db import get_recent_conversation_history, save_message
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="dsa_notes",
    embedding_function=embedding_fn
)

all_data = collection.get()
sources = [meta["source"] for meta in all_data["metadatas"]]

# Count chunks per source file
counts = Counter(sources)

print(f"Total chunks in database: {len(sources)}\n")
print("Chunks per file:")
for source, count in sorted(counts.items()):
    print(f"  {count} chunks — {source}")

# 1. Get recent conversation context (past 6 hours)
chat_history = get_recent_conversation_history()

# 2. Append the current user question to the history array
chat_history.append({"role": "user", "content": user_input})

# ... (Send chat_history to your LLM API and get `ai_response`) ...

# 3. Save both messages to SQLite using standard commas/strings
save_message("user",user_input)
save_message("assistant", ai_response)