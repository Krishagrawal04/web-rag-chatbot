# 🌐 LLM-Powered Web RAG Chatbot

## ✅ Phase 4 — FAISS + Embeddings
- Added `build_vectorstore()` using Google Gemini `gemini-embedding-001`
- Stores vectors in FAISS for fast semantic similarity search
- Batch processing (45 chunks/batch) with rate-limit handling
- API keys loaded securely from `.env` via python-dotenv

## 📌 Roadmap
- ✅ Phase 1 — Working Streamlit UI
- ✅ Phase 2 — Website scraping
- ✅ Phase 3 — Chunking
- ✅ Phase 4 — FAISS + Embeddings
- ⬜ Phase 5 — Groq integration
- ⬜ Phase 6 — Conversation memory

## 🔑 Setup
```bash
cp .env.example .env
# Add your GEMINI_API_KEY to .env
pip install -r requirements.txt
streamlit run app.py
```