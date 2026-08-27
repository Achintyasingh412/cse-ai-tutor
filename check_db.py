import chromadb
from chromadb.utils import embedding_functions

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
from collections import Counter
counts = Counter(sources)

print(f"Total chunks in database: {len(sources)}\n")
print("Chunks per file:")
for source, count in sorted(counts.items()):
    print(f"  {count} chunks — {source}")