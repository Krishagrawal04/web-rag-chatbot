# 🌐 LLM-Powered Web RAG Chatbot

A production-grade Retrieval-Augmented Generation (RAG) chatbot that enables real-time natural language Q&A over **any website**.

## ✅ Phase 6 — Conversation Memory (Final)
- Replaced single-turn `ask_question()` with `ConversationalRetrievalChain`
- Added `ConversationBufferMemory` — preserves full chat history across queries
- Bot now understands follow-up questions in context of previous answers
- Project complete and fully functional

## 🛠️ Tech Stack
| Component | Technology |
|---|---|
| Web Scraping | BeautifulSoup4 |
| Embeddings | Google Gemini (`gemini-embedding-001`) |
| Vector Store | FAISS |
| LLM | Groq API — LLaMA3.3-70B |
| RAG Chain + Memory | LangChain ConversationalRetrievalChain |
| UI | Streamlit |

## 📌 All Phases Complete
- ✅ Phase 1 — Working Streamlit UI
- ✅ Phase 2 — Website scraping
- ✅ Phase 3 — Chunking
- ✅ Phase 4 — FAISS + Embeddings
- ✅ Phase 5 — Groq integration
- ✅ Phase 6 — Conversation memory

## 🚀 How to Run
```bash
git clone https://github.com/YOUR_USERNAME/web-rag-chatbot.git
cd web-rag-chatbot
python -m venv rag_env
rag_env\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
streamlit run app.py
```

## 🔑 Getting API Keys
- **Groq API** (free): https://console.groq.com
- **Gemini API** (free): https://aistudio.google.com

## 📁 Project Structure
```
web-rag-chatbot/
├── app.py            # Streamlit UI
├── rag_engine.py     # Core RAG logic
├── requirements.txt  # Dependencies
├── .env.example      # API key template (safe to commit)
├── .env              # Your actual keys (git ignored)
├── .gitignore
└── README.md
```