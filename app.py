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
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional finish
st.markdown("""
<style>
    /* Sleek top border accent */
    .stApp > header {
        border-top: 4px solid #1E3A8A;
    }
    /* Style the expander title */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.title("🖥️ Enterprise IT Hardware Support")
st.markdown("**Domain-Specific RAG Chatbot | ITCC508 Lab Project**")

with st.expander("ℹ️ About this Intelligent Assistant & Experimental Modes", expanded=False):
    st.markdown("""
    Welcome to the **IT Hardware Helpdesk**. This Retrieval-Augmented Generation (RAG) assistant helps technicians instantly locate technical specifications, diagnostic codes, and repair procedures.
    
    **🧪 Experimental Modes (Use the Sidebar to test):**
    - **🟢 Strictly Grounded:** The standard RAG pipeline. It has `temperature=0` and a strict prompt. It will safely say "I cannot answer..." if you ask an off-topic question.
    - **🔴 Hallucination Stress-Test:** Bypasses safety guardrails (`temperature=1.0` and no strict prompt). Try asking it for a sourdough recipe to see it hallucinate!
    - **🟡 Baseline LLM:** Completely ignores the PDF manuals and acts like standard ChatGPT, relying only on its pre-trained knowledge.
    """)

st.divider()

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
    
    # 4. LLMs
    llm_safe = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0)
    llm_unsafe = ChatGroq(model_name="openai/gpt-oss-20b", temperature=1.0)
    
    # 5. Prompts
    prompt_grounded = ChatPromptTemplate.from_messages([
        ("system", "You are a specialized enterprise IT hardware support assistant.\nAnswer questions strictly using ONLY the provided hardware manual context below.\nIf the answer cannot be found in the context, reply: 'I cannot answer based on the provided domain data.'\nProvide your answers in a clean, professional, and easy-to-read format.\n\nContext:\n{context}"),
        ("human", "{input}")
    ])
    
    prompt_hallucinate = ChatPromptTemplate.from_messages([
        ("system", "You are a highly creative and confident assistant. Answer the user's question using your own general knowledge. You do not need to stick to the provided context. If you don't know the answer, confidently make one up (hallucinate) in detail.\n\nContext:\n{context}"),
        ("human", "{input}")
    ])

    prompt_baseline = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer the user's question using your pre-trained knowledge. Do not use the provided context.\n\nContext:\n{context}"),
        ("human", "{input}")
    ])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    class ExperimentalRagPipeline:
        def __init__(self, retriever):
            self.retriever = retriever

        def invoke(self, query, mode):
            if mode == "🟢 Strictly Grounded (Safe RAG)":
                active_llm = llm_safe
                active_prompt = prompt_grounded
                use_retrieval = True
            elif mode == "🔴 Hallucination Stress-Test":
                active_llm = llm_unsafe
                active_prompt = prompt_hallucinate
                use_retrieval = True
            else:
                active_llm = llm_safe
                active_prompt = prompt_baseline
                use_retrieval = False

            docs = self.retriever.invoke(query) if use_retrieval else []
            formatted_context = format_docs(docs) if docs else "No context provided."
            
            chain = active_prompt | active_llm | StrOutputParser()
            response_text = chain.invoke({"context": formatted_context, "input": query})
            
            return {"answer": response_text, "context": docs}

    return ExperimentalRagPipeline(retriever), "Success"

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ System Status")
    
    if "GROQ_API_KEY" in st.secrets:
        api_key_input = st.secrets["GROQ_API_KEY"]
        st.success("✅ Secure Cloud Connection: Active")
    else:
        api_key_input = st.text_input("Groq API Key", type="password")
        
    st.divider()
    
    st.header("🧪 Experimental Modes")
    selected_mode = st.radio(
        "Select AI Behavior:",
        [
            "🟢 Strictly Grounded (Safe RAG)",
            "🔴 Hallucination Stress-Test",
            "🟡 Baseline LLM (No RAG Context)"
        ],
        index=0
    )
    
    st.divider()
    
    st.markdown("### 📊 Architecture Stack")
    st.markdown("- **LLM Engine:** `gpt-oss-20b`")
    st.markdown("- **Embeddings:** `all-MiniLM-L6-v2`")
    st.markdown("- **Vector DB:** `ChromaDB`")
    
    st.divider()
    
    st.markdown("### 👨‍💻 Developer Info")
    st.markdown("**Developer:** John Loyd Arcilla")
    st.markdown("**Course:** ITCC508 Lab PT-M1")

# --- Main Chat Interface ---
if api_key_input:
    with st.spinner("⚙️ Initializing IT Support Engine..."):
        rag_chain, status_msg = init_rag_pipeline(api_key_input)
        
    if rag_chain is None:
        st.error(status_msg)
    else:
        # Helper function to get the correct avatar
        def get_avatar(role):
            return "👤" if role == "user" else "🖥️"

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=get_avatar(message["role"])):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input(f"Ask a hardware question ({selected_mode.split(' ')[0]} Mode)..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar="🖥️"):
                message_placeholder = st.empty()
                with st.spinner(f"Processing in {selected_mode.split(' ')[1]} mode..."):
                    try:
                        response = rag_chain.invoke(prompt, selected_mode)
                        answer = response["answer"]
                        
                        citations = ""
                        if response['context']:
                            citations = "\n\n---\n**📚 Source Documents Used:**\n"
                            for i, doc in enumerate(response['context']):
                                source = os.path.basename(doc.metadata.get('source', 'Unknown'))
                                page = doc.metadata.get('page', 'Unknown')
                                citations += f"- `{source}` (Page {page})\n"
                        elif selected_mode == "🟡 Baseline LLM (No RAG Context)":
                            citations = "\n\n---\n*⚠️ No PDF manuals were referenced (Baseline Mode).*"
                        elif selected_mode == "🔴 Hallucination Stress-Test":
                            citations = "\n\n---\n*⚠️ Context was retrieved, but guardrails were removed. Output may be hallucinated.*"
                            
                        full_response = answer + citations
                        message_placeholder.markdown(full_response)
                        
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
else:
    st.info("👈 Please configure your connection in the sidebar.")
