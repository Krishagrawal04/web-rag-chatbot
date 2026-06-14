# 🌐 LLM-Powered Web RAG Chatbot

A production-grade Retrieval-Augmented Generation (RAG) chatbot that enables real-time natural language Q&A over **any website**.

## 🛠️ Tech Stack
| Component | Technology |
|---|---|
| Web Scraping | BeautifulSoup4 |
| Embeddings | Google Gemini (`gemini-embedding-001`) |
| Vector Store | FAISS |
| LLM | Groq API — LLaMA3.3-70B |
| RAG Chain + Memory | LangChain ConversationalRetrievalChain |
| UI | Streamlit |

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/Krishagrawal04/web-rag-chatbot.git
cd web-rag-chatbot
```

### 2. Create virtual environment
```bash
python -m venv rag_env
rag_env\Scripts\activate        # Windows
# source rag_env/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys
```bash
cp .env.example .env
# Open .env and add your API keys
```

### 5. Run the app
```bash
streamlit run app.py
```

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

## 🔑 Getting API Keys
- **Groq API** (free): https://console.groq.com
- **Gemini API** (free): https://aistudio.google.com

## 📌 All Phases Complete
- ✅ Phase 1 — Working Streamlit UI
- ✅ Phase 2 — Website scraping
- ✅ Phase 3 — Chunking
- ✅ Phase 4 — FAISS + Embeddings
- ✅ Phase 5 — Groq integration + Conversation Memory
