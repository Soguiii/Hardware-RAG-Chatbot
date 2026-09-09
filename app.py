import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- UI Configuration (Professional Soft Theme) ---
st.set_page_config(page_title="Hardware Support Chatbot", page_icon="🖥️", layout="centered")



st.title("🖥️ IT Helpdesk Chatbot")
st.caption("A Domain-Specific RAG system for Dell & HP Hardware Manuals")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your technical support assistant. How can I help you with your hardware today?"}
    ]

# --- RAG Pipeline Initialization ---
@st.cache_resource(show_spinner=False)
def init_rag_pipeline(api_key):
    os.environ["GROQ_API_KEY"] = api_key
    
    # 1. Load Data (Searches current directory and subdirectories for PDFs)
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
        "You are a specialized AI assistant for the user's uploaded domain.\n"
        "Answer questions strictly using ONLY the provided context below.\n"
        "If the answer cannot be found in the context, reply: 'I cannot answer based on the provided domain data.'\n\n"
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
    st.header("⚙️ Configuration")
    api_key_input = st.text_input("Groq API Key", type="password", help="Enter your Groq API key to activate the chatbot.")
    st.markdown("---")
    st.markdown("**Domain:** IT Helpdesk & Hardware")
    st.markdown("**LLM:** `openai/gpt-oss-20b`")
    st.markdown("**Embeddings:** `all-MiniLM-L6-v2`")

# --- Main Chat Interface ---
if api_key_input:
    with st.spinner("⚙️ Initializing RAG Pipeline (Loading PDFs and generating embeddings...)"):
        rag_chain, status_msg = init_rag_pipeline(api_key_input)
        
    if rag_chain is None:
        st.error(status_msg)
    else:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Ask a question about the hardware manuals..."):
            # Add user message to state and display
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate AI response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Searching manuals..."):
                    try:
                        response = rag_chain.invoke({"input": prompt})
                        answer = response["answer"]
                        
                        # Format source citations
                        citations = "\n\n---\n**📚 Source Documents:**\n"
                        for i, doc in enumerate(response['context']):
                            source = doc.metadata.get('source', 'Unknown')
                            # Clean up the path for display
                            source_name = os.path.basename(source)
                            citations += f"- `{source_name}`\n"
                            
                        full_response = answer + citations
                        message_placeholder.markdown(full_response)
                        
                        # Save to state
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
else:
    st.info("👈 Please enter your Groq API Key in the sidebar to start the chatbot.")
