import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- UI Configuration ---
st.set_page_config(
    page_title="IT Helpdesk | Hardware Support",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Header Section ---
st.title("🛠️ Enterprise IT Hardware Support")
st.markdown("**Domain-Specific RAG Chatbot | ITCC508 Lab Project**")

# Information Expander
with st.expander("ℹ️ About this Intelligent Assistant", expanded=True):
    st.markdown("""
    Welcome to the **IT Hardware Helpdesk**. This Retrieval-Augmented Generation (RAG) assistant helps technicians instantly locate exact technical specifications, diagnostic LED/beep codes, and component replacement procedures without manually sifting through hundreds of pages.
    
    **📚 Currently Supported Hardware Data:**
    - 🖥️ **Dell OptiPlex 7080 SFF** (Official Service Manual)
    - 💻 **HP All-in-One Desktop** (Hardware Reference Guide)
    - 🖧 **HP EliteDesk 805 G6 SFF** (Maintenance & Service Guide)
    
    💡 **Example Queries to Try:**
    - *"What does a blinking amber power LED mean on the Dell OptiPlex?"*
    - *"How do I clear the CMOS or reset the BIOS?"*
    - *"What are the exact steps to replace an M.2 NVMe SSD?"*
    """)

st.markdown("---")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome! All hardware manuals are loaded. What IT issue can I help you troubleshoot today?"}
    ]

# --- RAG Pipeline Initialization ---
@st.cache_resource(show_spinner=False)
def init_rag_pipeline(api_key):
    os.environ["GROQ_API_KEY"] = api_key
    
    # 1. Load Data
    loader = DirectoryLoader(".", glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=False)
    raw_documents = loader.load()
    
    if not raw_documents:
        return None, "No PDF manuals found. Please place your PDFs in this directory."
        
    # 2. Chunk Data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
    documents = text_splitter.split_documents(raw_documents)
    
    # 3. Embeddings & VectorDB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 4. LLM & Prompt
    llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0)
    system_prompt = (
        "You are a specialized enterprise IT hardware support assistant.\n"
        "Answer questions strictly using ONLY the provided hardware manual context below.\n"
        "If the answer cannot be found in the context, reply: 'I cannot answer based on the provided domain data.'\n"
        "Provide your answers in a clean, professional, and easy-to-read format (use bullet points for steps).\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 5. Pipeline Assembly
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    class RagPipeline:
        def __init__(self, retriever, prompt, llm):
            self.retriever = retriever
            self.prompt = prompt
            self.llm = llm

        def invoke(self, inputs):
            query = inputs["input"]
            docs = self.retriever.invoke(query)
            formatted_context = format_docs(docs)
            response_text = (self.prompt | self.llm | StrOutputParser()).invoke({
                "context": formatted_context,
                "input": query
            })
            return {"answer": response_text, "context": docs}

    return RagPipeline(retriever, prompt, llm), "Success"

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ System Status")
    
    # Secure API Key Loading
    if "GROQ_API_KEY" in st.secrets:
        api_key_input = st.secrets["GROQ_API_KEY"]
        st.success("✅ Secure Cloud Connection: Active")
    else:
        api_key_input = st.text_input("Groq API Key", type="password", help="Enter your Groq API key to activate the chatbot.")
        st.warning("⚠️ Running locally (No cloud secrets found)")
        
    st.markdown("---")
    st.markdown("### 📊 Architecture")
    st.markdown("- **LLM Engine:** `openai/gpt-oss-20b` (via Groq LPU)")
    st.markdown("- **Embeddings:** `all-MiniLM-L6-v2`")
    st.markdown("- **Vector Database:** `ChromaDB`")
    st.markdown("- **Chunking Strategy:** 700 chars / 70 overlap")
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer Info")
    st.markdown("**Developer:** Soji")
    st.markdown("**Project:** ITCC508 Lab PT-M1")
    st.markdown("**Course:** Intro to LLMOps and RAG Concepts")

# --- Main Chat Interface ---
if api_key_input:
    with st.spinner("⚙️ Initializing Engine (Reading manuals and loading embeddings...)"):
        rag_chain, status_msg = init_rag_pipeline(api_key_input)
        
    if rag_chain is None:
        st.error(status_msg)
    else:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Ask a technical question about the Dell or HP hardware..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate AI response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Searching hardware manuals..."):
                    try:
                        response = rag_chain.invoke({"input": prompt})
                        answer = response["answer"]
                        
                        # Format source citations cleanly
                        citations = "\n\n---\n**📚 Source Documents Used:**\n"
                        for i, doc in enumerate(response['context']):
                            source = doc.metadata.get('source', 'Unknown')
                            source_name = os.path.basename(source)
                            page = doc.metadata.get('page', 'Unknown')
                            citations += f"- `{source_name}` (Page {page})\n"
                            
                        full_response = answer + citations
                        message_placeholder.markdown(full_response)
                        
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
else:
    st.info("👈 Please configure your connection in the sidebar to boot the system.")
