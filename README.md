# 🌐 LLM-Powered Web RAG Chatbot

## ✅ Phase 3 — Chunking
- Added `chunk_text()` using LangChain's `RecursiveCharacterTextSplitter`
- chunk_size=1500, chunk_overlap=150 for context preservation
- Capped at 80 chunks to respect Gemini free-tier rate limits
- UI now shows chunk count metric

## 📌 Roadmap
- ✅ Phase 1 — Working Streamlit UI
- ✅ Phase 2 — Website scraping
- ✅ Phase 3 — Chunking
- ⬜ Phase 4 — FAISS + Embeddings
- ⬜ Phase 5 — Groq integration
- ⬜ Phase 6 — Conversation memory

## 🚀 Run
```bash
pip install -r requirements.txt
streamlit run app.py
```