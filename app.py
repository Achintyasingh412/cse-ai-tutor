import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# On Streamlit Cloud, secrets come from st.secrets, not a .env file — bridge them here
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

@st.cache_resource
def load_resources():
    client = genai.Client()
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(
        name="dsa_notes",
        embedding_function=embedding_fn
    )
    return client, collection

client, collection = load_resources()

SYSTEM_INSTRUCTION = (
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
)

def retrieve_context(question, n_results=5):
    results = collection.query(query_texts=[question], n_results=n_results)
    return "\n\n---\n\n".join(results["documents"][0])

def get_answer_stream(question):
    context = retrieve_context(question)
    stream = client.interactions.create(
        model="gemini-3.5-flash",
        system_instruction=SYSTEM_INSTRUCTION,
        input=f"CONTEXT:\n{context}\n\nQUESTION:\n{question}",
        stream=True,
    )
    for event in stream:
        if event.event_type == "step.delta" and event.delta:
            if getattr(event.delta, "type", None) == "text" and getattr(event.delta, "text", None):
                yield event.delta.text

# ---- Streamlit UI ----
st.set_page_config(page_title="Luna — CSE AI Tutor", page_icon="🌙")

with st.sidebar:
    st.header("🌙 Luna")
    st.caption("Your personal CSE tutor")
    st.divider()
    st.subheader("Topics loaded")
    st.markdown("- DSA (Arrays, Linked Lists, Stacks/Queues, Trees, Sorting, Graphs, Hashing, Recursion, DP, Greedy/Heaps, Complexity, Strings/Bits)")
    st.markdown("- OOP Fundamentals & SOLID Principles")
    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.title("🌙 Luna — CSE AI Tutor")
st.caption("Ask me anything from your DSA and OOP notes.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask Luna a question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        answer = st.write_stream(get_answer_stream(user_input))
    st.session_state.messages.append({"role": "assistant", "content": answer})