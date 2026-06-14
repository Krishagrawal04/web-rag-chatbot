import os
import time
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

load_dotenv()


def scrape_website(url: str) -> str:
    """Scrape and clean text content from any public website URL."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
    except Exception as e:
        raise RuntimeError(f"Failed to scrape URL: {e}")


def chunk_text(text: str, url: str) -> list:
    """Split scraped text into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)[:80]
    return [Document(page_content=chunk, metadata={"source": url}) for chunk in chunks]


def build_vectorstore(docs: list, gemini_api_key: str) -> FAISS:
    """Embed documents with Gemini and store in FAISS.
    Processes in batches of 45 to respect Gemini free-tier rate limit (100 req/min).
    """
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    BATCH_SIZE = 45
    vectorstore = None
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(documents=batch, embedding=embeddings)
        else:
            vectorstore.add_documents(batch)
        if i + BATCH_SIZE < len(docs):
            time.sleep(65)
    return vectorstore


def build_qa_chain(vectorstore: FAISS, groq_api_key: str) -> ConversationalRetrievalChain:
    """Build a conversational RAG chain with multi-turn memory using Groq LLM."""
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
    )
    # ConversationBufferMemory preserves full chat history for context-aware responses
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )