import os
import streamlit as st
from dotenv import load_dotenv
from rag_engine import (
    scrape_website,
    chunk_text,
    build_vectorstore,
    build_qa_chain,
)

load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WebRAG Chatbot",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    .stApp { background: #0a0a0f; color: #e8e8f0; }

    [data-testid="stSidebar"] {
        background: #0f0f1a !important;
        border-right: 1px solid #1e1e3a;
    }
    [data-testid="stSidebar"] * { color: #c8c8e0 !important; }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 2.6rem;
        background: linear-gradient(135deg, #7c6ff7, #f76fa8, #6fcff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.15;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        font-family: 'Space Mono', monospace;
        font-size: 0.78rem;
        color: #6060a0;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 1.8rem;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .status-ready   { background: #0d2b1e; color: #3effa0; border: 1px solid #1a5c38; }
    .status-loading { background: #1e1a0d; color: #ffcc3e; border: 1px solid #5c4a1a; }
    .status-idle    { background: #1a1a2e; color: #7878b8; border: 1px solid #2a2a5a; }

    .chat-user {
        background: #12122a;
        border: 1px solid #2a2a55;
        border-radius: 14px 14px 4px 14px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.95rem;
        color: #d8d8f5;
        max-width: 80%;
        margin-left: auto;
    }
    .chat-bot {
        background: #0f1a2a;
        border: 1px solid #1a3055;
        border-radius: 14px 14px 14px 4px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.95rem;
        color: #c8d8f0;
        max-width: 88%;
        line-height: 1.65;
    }
    .chat-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 6px;
        opacity: 0.55;
    }
    .stTextInput > div > div > input {
        background: #12122a !important;
        border: 1px solid #2a2a55 !important;
        border-radius: 10px !important;
        color: #e8e8f8 !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 12px 16px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #7c6ff7 !important;
        box-shadow: 0 0 0 2px rgba(124,111,247,0.15) !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7c6ff7, #a06ff7) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        padding: 10px 28px !important;
        font-size: 0.9rem !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #9a8ff9, #bc8ff9) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(124,111,247,0.35) !important;
    }
    hr { border-color: #1e1e3a !important; }
    [data-testid="stMetric"] {
        background: #0f0f1e;
        border: 1px solid #1e1e3a;
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"] {
        color: #7c6ff7 !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 1.4rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #6060a0 !important;
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    </style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
for key, default in {
    "chat_history": [],
    "qa_chain": None,
    "current_url": "",
    "doc_count": 0,
    "status": "idle",
    "last_question": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌐 WebRAG Chatbot")
    st.markdown("---")
    st.markdown("**Enter a Website URL**")
    url_input = st.text_input("URL", placeholder="https://example.com",
                               label_visibility="collapsed", key="url_input")
    load_btn = st.button("🚀 Load & Index Website", use_container_width=True)
    st.markdown("---")

    if st.session_state.status == "ready":
        st.markdown('<span class="status-badge status-ready">● READY</span>', unsafe_allow_html=True)
        url_display = st.session_state.current_url
        st.caption(f"Indexed: `{url_display[:40]}...`" if len(url_display) > 40 else f"Indexed: `{url_display}`")
    elif st.session_state.status == "loading":
        st.markdown('<span class="status-badge status-loading">⟳ LOADING</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-idle">○ IDLE</span>', unsafe_allow_html=True)

    st.markdown("---")
    if st.session_state.status == "ready":
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks", st.session_state.doc_count)
        with col2:
            st.metric("Turns", len([m for m in st.session_state.chat_history if m["role"] == "user"]))

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.qa_chain = None
        st.session_state.current_url = ""
        st.session_state.doc_count = 0
        st.session_state.status = "idle"
        st.session_state.last_question = ""
        st.rerun()

    st.markdown("---")
    st.markdown("""
        <div style='font-family:Space Mono,monospace;font-size:0.68rem;color:#404070;line-height:1.8'>
        STACK<br>
        ├─ LangChain RAG<br>
        ├─ Groq LLaMA3.3-70B<br>
        ├─ Gemini Embeddings<br>
        ├─ FAISS Vector Store<br>
        └─ BeautifulSoup4
        </div>
    """, unsafe_allow_html=True)

# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">WebRAG Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Real-time Q&A over any website · Powered by LangChain + Groq + Gemini</div>', unsafe_allow_html=True)

if not GROQ_API_KEY or not GEMINI_API_KEY:
    st.error("⚠️ API keys not found! Please create a `.env` file from `.env.example` and add your keys.")
    st.stop()

# ── Load Website ──────────────────────────────────────────────────────────────
if load_btn and url_input:
    if not url_input.startswith("http"):
        st.error("Please enter a valid URL starting with http:// or https://")
    else:
        st.session_state.status = "loading"
        st.session_state.chat_history = []
        st.session_state.qa_chain = None
        st.session_state.last_question = ""

        with st.spinner("🔍 Scraping website..."):
            try:
                raw_text = scrape_website(url_input)
                st.toast("✅ Website scraped!", icon="🌐")
            except Exception as e:
                st.error(f"Scraping failed: {e}")
                st.session_state.status = "idle"
                st.stop()

        with st.spinner("✂️ Chunking content..."):
            docs = chunk_text(raw_text, url_input)
            st.toast(f"✅ Created {len(docs)} chunks!", icon="✂️")

        with st.spinner("🔢 Generating embeddings & indexing with FAISS..."):
            try:
                vectorstore = build_vectorstore(docs, GEMINI_API_KEY)
                st.toast("✅ Vectors stored in FAISS!", icon="🗄️")
            except Exception as e:
                st.error(f"Embedding failed: {e}")
                st.session_state.status = "idle"
                st.stop()

        with st.spinner("🤖 Initialising Groq LLM chain..."):
            try:
                qa_chain = build_qa_chain(vectorstore, GROQ_API_KEY)
                st.session_state.qa_chain = qa_chain
                st.session_state.current_url = url_input
                st.session_state.doc_count = len(docs)
                st.session_state.status = "ready"
                st.toast("🚀 Ready to chat!", icon="🤖")
            except Exception as e:
                st.error(f"LLM chain failed: {e}")
                st.session_state.status = "idle"
                st.stop()

        st.rerun()

# ── Chat Interface ────────────────────────────────────────────────────────────
if st.session_state.status != "ready":
    st.markdown("""
        <div style='margin-top:3rem;text-align:center;padding:3rem;
                    background:#0d0d1e;border:1px dashed #2a2a55;border-radius:16px;'>
            <div style='font-size:2.5rem;margin-bottom:1rem'>🌐</div>
            <div style='font-family:Syne,sans-serif;font-size:1.1rem;color:#5050a0;'>
                Enter a URL in the sidebar and click<br>
                <strong style='color:#7c6ff7'>Load & Index Website</strong> to begin
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user"><div class="chat-label">You</div>{msg["content"]}</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="chat-bot"><div class="chat-label">WebRAG · LLaMA3.3-70B</div>{msg["content"]}</div>',
                unsafe_allow_html=True)

    st.markdown("---")

    with st.form(key="chat_form", clear_on_submit=True):
        col_q, col_btn = st.columns([5, 1])
        with col_q:
            question = st.text_input("Ask a question",
                placeholder="What is this website about? Ask anything...",
                label_visibility="collapsed")
        with col_btn:
            submitted = st.form_submit_button("Send →", use_container_width=True)

    if submitted and question.strip():
        user_q = question.strip()
        st.session_state.chat_history.append({"role": "user", "content": user_q})

        with st.spinner("🧠 Thinking..."):
            try:
                result = st.session_state.qa_chain.invoke({"question": user_q})
                answer = result.get("answer", "Sorry, I couldn't find a relevant answer.")
                sources = result.get("source_documents", [])
            except Exception as e:
                answer = f"Error generating response: {e}"
                sources = []

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        if sources:
            with st.expander("📄 Retrieved source chunks"):
                for i, doc in enumerate(sources[:3], 1):
                    st.markdown(
                        f"**Chunk {i}** — `{doc.metadata.get('source', 'unknown')}`\n\n{doc.page_content[:300]}...")

        st.rerun()