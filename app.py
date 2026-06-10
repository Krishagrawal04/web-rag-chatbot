import os
import streamlit as st
from dotenv import load_dotenv
from rag_engine import scrape_website, chunk_text, build_vectorstore

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="WebRAG Chatbot", page_icon="🌐", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    .stApp { background: #0a0a0f; color: #e8e8f0; }
    [data-testid="stSidebar"] { background: #0f0f1a !important; border-right: 1px solid #1e1e3a; }
    [data-testid="stSidebar"] * { color: #c8c8e0 !important; }
    .hero-title {
        font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.6rem;
        background: linear-gradient(135deg, #7c6ff7, #f76fa8, #6fcff7);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; line-height: 1.15; margin-bottom: 0.2rem;
    }
    .hero-sub { font-family: 'Space Mono', monospace; font-size: 0.78rem; color: #6060a0;
        letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 1.8rem; }
    .status-badge { display: inline-block; padding: 4px 14px; border-radius: 20px;
        font-family: 'Space Mono', monospace; font-size: 0.72rem; font-weight: 700; margin-bottom: 1rem; }
    .status-ready   { background: #0d2b1e; color: #3effa0; border: 1px solid #1a5c38; }
    .status-loading { background: #1e1a0d; color: #ffcc3e; border: 1px solid #5c4a1a; }
    .status-idle    { background: #1a1a2e; color: #7878b8; border: 1px solid #2a2a5a; }
    .stButton > button {
        background: linear-gradient(135deg, #7c6ff7, #a06ff7) !important;
        color: #fff !important; border: none !important; border-radius: 10px !important;
        font-family: 'Syne', sans-serif !important; font-weight: 600 !important;
        padding: 10px 28px !important; font-size: 0.9rem !important;
    }
    hr { border-color: #1e1e3a !important; }
    [data-testid="stMetric"] { background: #0f0f1e; border: 1px solid #1e1e3a; border-radius: 10px; padding: 12px 16px; }
    [data-testid="stMetricValue"] { color: #7c6ff7 !important; font-family: 'Space Mono', monospace !important; font-size: 1.4rem !important; }
    [data-testid="stMetricLabel"] { color: #6060a0 !important; font-size: 0.72rem !important; text-transform: uppercase; }
    </style>
""", unsafe_allow_html=True)

for key, default in {
    "status": "idle", "current_url": "", "doc_count": 0, "vectorstore": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.markdown("### 🌐 WebRAG Chatbot")
    st.markdown("---")
    st.markdown("**Enter a Website URL**")
    url_input = st.text_input("URL", placeholder="https://example.com",
                               label_visibility="collapsed", key="url_input")
    load_btn = st.button("🚀 Load & Index Website", use_container_width=True)
    st.markdown("---")
    if st.session_state.status == "ready":
        st.markdown('<span class="status-badge status-ready">● INDEXED</span>', unsafe_allow_html=True)
        st.metric("Chunks", st.session_state.doc_count)
    elif st.session_state.status == "loading":
        st.markdown('<span class="status-badge status-loading">⟳ LOADING</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-idle">○ IDLE</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
        <div style='font-family:Space Mono,monospace;font-size:0.68rem;color:#404070;line-height:1.8'>
        STACK<br>├─ LangChain RAG<br>├─ Groq LLaMA3.3-70B<br>
        ├─ Gemini Embeddings<br>├─ FAISS Vector Store<br>└─ BeautifulSoup4
        </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="hero-title">WebRAG Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Real-time Q&A over any website · Powered by LangChain + Groq + Gemini</div>', unsafe_allow_html=True)

if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY not found! Please create a `.env` file from `.env.example`.")
    st.stop()

if load_btn and url_input:
    if not url_input.startswith("http"):
        st.error("Please enter a valid URL starting with http:// or https://")
    else:
        st.session_state.status = "loading"
        with st.spinner("🔍 Scraping website..."):
            try:
                text = scrape_website(url_input)
                st.toast("✅ Website scraped!", icon="🌐")
            except Exception as e:
                st.error(f"Scraping failed: {e}")
                st.session_state.status = "idle"
                st.stop()
        with st.spinner("✂️ Chunking content..."):
            docs = chunk_text(text, url_input)
            st.toast(f"✅ Created {len(docs)} chunks!", icon="✂️")
        with st.spinner("🔢 Generating embeddings & indexing with FAISS..."):
            try:
                vectorstore = build_vectorstore(docs, GEMINI_API_KEY)
                st.session_state.vectorstore = vectorstore
                st.session_state.doc_count = len(docs)
                st.session_state.current_url = url_input
                st.session_state.status = "ready"
                st.toast("✅ Vectors stored in FAISS!", icon="🗄️")
            except Exception as e:
                st.error(f"Embedding failed: {e}")
                st.session_state.status = "idle"
                st.stop()
        st.rerun()

if st.session_state.status == "ready":
    st.success(f"✅ `{st.session_state.current_url}` — {st.session_state.doc_count} chunks indexed in FAISS. Ready for Groq integration in Phase 5!")
else:
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