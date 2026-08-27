# 🌙 Luna — CSE AI Tutor

A RAG-based (Retrieval-Augmented Generation) AI chatbot that answers Computer Science questions grounded in a curated knowledge base of DSA and OOP notes. Built as a first hands-on AI/ML project to learn embeddings, vector search, and LLM-powered applications end to end.

**Live demo:** [cse-ai-tutor-wnnptpyzkym3hm6eiavuaf.streamlit.app](https://cse-ai-tutor-wnnptpyzkym3hm6eiavuaf.streamlit.app)

## What it does

Luna answers CSE questions (currently DSA and OOP fundamentals) using a Retrieval-Augmented Generation pipeline:

1. A knowledge base of topic notes is chunked and converted into vector embeddings.
2. When a question is asked, the most relevant chunks are retrieved via semantic search.
3. Those chunks are passed to Google's Gemini model as context, so answers are grounded in the actual notes rather than relying purely on the model's general training knowledge.
4. Responses stream in live and are formatted as short bullet points, tuned for quick exam/interview revision.

## Tech Stack

- **LLM:** Google Gemini (`gemini-3.5-flash`) via the Gemini API
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector Database:** ChromaDB
- **PDF Parsing:** `pypdf`
- **Frontend:** Streamlit
- **Deployment:** Streamlit Community Cloud

## How it works
knowledge/*.pdf → chunked text → embeddings → ChromaDB
│
user question ──► embed question ──► search ChromaDB ──► top-k relevant chunks
│
chunks + question ──► Gemini ──► streamed answer
## Project structure
├── app.py # Streamlit frontend (main app)
├── chatbot.py # Terminal version of the chatbot
├── ingest.py # Builds the vector database from PDFs in /knowledge
├── check_db.py # Debug utility to inspect the vector database
├── knowledge/ # Source PDFs (DSA, OOP topics)
├── requirements.txt
└── .env # API key (not committed)


## Running it locally

```bash
git clone https://github.com/jeeveshcodes/cse-ai-tutor.git
cd cse-ai-tutor
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file with:

GEMINI_API_KEY=your_key_here


Build the knowledge base, then run the app:
```bash
python ingest.py
streamlit run app.py
```

## Roadmap

- [ ] Add DBMS, OS, and CN topic notes
- [ ] Voice input support
- [ ] Expand test coverage across more question types

---

Built by [Jeevesh](https://github.com/jeeveshcodes) as a first AI/ML project, alongside coursework at SRM KTR (CSE Core).