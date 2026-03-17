import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# --- STABLE IMPORTS ---
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ❌ removed old memory import

# 1. LOAD ENV & PATHS
load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
INDEX_PATH = os.getenv("INDEX_PATH", "faiss_index")
PDF_STORAGE_PATH = os.getenv("PDF_STORAGE_PATH", "uploaded_pdfs")

for path in [PDF_STORAGE_PATH, INDEX_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)

st.set_page_config(page_title="Enterprise RAG v2", layout="wide")
st.title("🛡️ Local PDF Intelligence (Chat Mode)")

# ✅ INIT CHAT HISTORY
st.session_state.setdefault("history", [])  # ensure history exists

# ✅ FUNCTION TO FORMAT HISTORY

def format_history():
    history = st.session_state.get("history", [])  # safe read
    return "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in history]
    )

# 2. MODELS (CACHED)
@st.cache_resource
def load_models():
    llm = ChatOllama(model=LLM_MODEL)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return llm, embeddings

llm, embeddings = load_models()

# 3. RETRIEVAL LOGIC
def get_retriever():
    if os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    return None

def process_pdfs(pdf_files):
    all_docs = []
    for pdf in pdf_files:
        file_path = os.path.join(PDF_STORAGE_PATH, pdf.name)
        with open(file_path, "wb") as f:
            f.write(pdf.getbuffer())
        reader = PdfReader(pdf)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                all_docs.append(Document(page_content=text, metadata={"source": pdf.name, "page": i + 1}))
    
    if not all_docs: return "No text found."
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    final_chunks = text_splitter.split_documents(all_docs)
    
    vdb = FAISS.from_documents(final_chunks, embeddings)
    vdb.save_local(INDEX_PATH)
    return "Database Synchronized!"

# 4. SIDEBAR UI
with st.sidebar:
    st.header("Control Panel")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("🚀 Sync Knowledge Base"):
        if uploaded_files:
            with st.spinner("Processing..."):
                st.toast(process_pdfs(uploaded_files))
            st.rerun()

# 5. CHAT INTERFACE (MULTI-TURN WITH MEMORY)
retriever = get_retriever()

if retriever:

    # ✅ DISPLAY OLD CHAT
    for msg in st.session_state.history:
        st.chat_message(msg["role"]).markdown(msg["content"])

    if query := st.chat_input("Ask me about the documents..."):

        # ✅ SAVE USER MESSAGE
        st.session_state.history.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)

        with st.chat_message("assistant"):

            # A. Retrieve Context
            context_docs = retriever.invoke(query)
            context_text = "\n\n".join(
                [f"Source: {d.metadata['source']} (P.{d.metadata['page']})\n{d.page_content}" for d in context_docs]
            )

            # B. PROMPT WITH HISTORY
            prompt = ChatPromptTemplate.from_template("""
            You are a helpful assistant.

            Chat History:
            {history}

            Context:
            {context}

            Question:
            {question}
            """)

            # C. CREATE CHAIN (with history)
            chain = (
                {
                    "context": lambda _: context_text,
                    "question": RunnablePassthrough(),
                    "history": lambda _: format_history(),  # inject memory
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            # D. STREAM RESPONSE
            placeholder = st.empty()
            full_response = ""

            for chunk in chain.stream(query):
                full_response += chunk
                placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

            # ✅ SAVE ASSISTANT RESPONSE
            st.session_state.history.append({"role": "assistant", "content": full_response})

            # E. SHOW CITATIONS
            with st.expander("🔍 Citations"):
                for d in context_docs:
                    st.write(f"- {d.metadata['source']} (Page {d.metadata['page']})")

else:
    st.warning("Upload and Sync PDFs to start.")