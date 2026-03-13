import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Load Environment Variables
load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
INDEX_PATH = os.getenv("INDEX_PATH", "faiss_index")
PDF_STORAGE_PATH = os.getenv("PDF_STORAGE_PATH", "uploaded_pdfs")

# Ensure directories exist
for path in [PDF_STORAGE_PATH, INDEX_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)

st.set_page_config(page_title="Enterprise RAG System", layout="wide")
st.title("🛡️ Advanced Local PDF Intelligence")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- MODELS ---
@st.cache_resource
def load_models():
    # Chat Model
    llm = ChatOllama(model=LLM_MODEL)
    # Using Ollama for Embeddings as requested
    embeddings = OllamaEmbeddings(model=EMBED_MODEL,temperature=0)
    return llm, embeddings

llm, embeddings = load_models()

# --- CORE LOGIC ---

def get_vectorstore():
    if os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

def process_pdfs(pdf_files):
    all_docs = []
    for pdf in pdf_files:
        file_path = os.path.join(PDF_STORAGE_PATH, pdf.name)
        if os.path.exists(file_path): continue
            
        with open(file_path, "wb") as f:
            f.write(pdf.getbuffer())
        
        reader = PdfReader(pdf)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                all_docs.append(Document(
                    page_content=text, 
                    metadata={"source": pdf.name, "page": i + 1}
                ))
    
    if not all_docs: return "No new files added."

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    final_chunks = text_splitter.split_documents(all_docs)
    
    existing_vdb = get_vectorstore()
    if existing_vdb:
        existing_vdb.add_documents(final_chunks)
        existing_vdb.save_local(INDEX_PATH)
    else:
        new_vdb = FAISS.from_documents(final_chunks, embeddings)
        new_vdb.save_local(INDEX_PATH)
    return "Database Synchronized!"

# --- UI & CHAT ---

with st.sidebar:
    st.header("Settings")
    st.info(f"LLM: {LLM_MODEL}\n\nEmbed: {EMBED_MODEL}")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("🚀 Sync Knowledge Base"):
        if uploaded_files:
            st.toast(process_pdfs(uploaded_files))
            st.rerun()

vectorstore = get_vectorstore()

if vectorstore:
    # Chat History
    for message in st.session_state.chat_history:
        with st.chat_message("Human" if isinstance(message, HumanMessage) else "AI"):
            st.markdown(message.content)

    query = st.chat_input("Ask about your documents...")
    if query:
        st.session_state.chat_history.append(HumanMessage(content=query))
        with st.chat_message("Human"): st.markdown(query)

        with st.chat_message("AI"):
            # RAG Flow
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            context_docs = retriever.invoke(query)
            context_text = "\n\n".join([f"Source: {d.metadata['source']} (P.{d.metadata['page']})\n{d.page_content}" for d in context_docs])

            prompt = ChatPromptTemplate.from_template("Answer using context:\n\n{context}\n\nQuestion: {question}")
            chain = prompt | llm | StrOutputParser()
            
            response = chain.invoke({"context": context_text, "question": query})
            st.markdown(response)
            
            with st.expander("🔍 Citations"):
                for d in context_docs:
                    st.write(f"- {d.metadata['source']} (Page {d.metadata['page']})")
            
            st.session_state.chat_history.append(AIMessage(content=response))
else:
    st.warning("Please upload PDFs and Sync to begin.")