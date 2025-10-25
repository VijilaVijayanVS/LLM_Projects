import streamlit as st
import os
from datetime import datetime
import time
import hashlib
from pathlib import Path
import tempfile

# Document processing
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
import torch

# Set page config
st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for pastel colors and enhanced UI
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5e6ff 0%, #ffe6f0 50%, #e6f3ff 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0e6ff 0%, #ffe6f5 100%);
    }
    
    /* Cards and containers */
    .stAlert, .stTextInput, .stTextArea {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Headers */
    h1 {
        color: #8b5cf6;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h2, h3 {
        color: #ec4899;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 15px;
        padding: 20px;
        border: 2px dashed #8b5cf6;
    }
    
    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
        color: white;
        padding: 15px;
        border-radius: 18px 18px 4px 18px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .assistant-message {
        background: rgba(255, 255, 255, 0.9);
        color: #1f2937;
        padding: 15px;
        border-radius: 18px 18px 18px 4px;
        margin: 10px 0;
        border-left: 4px solid #8b5cf6;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .system-message {
        background: rgba(147, 197, 253, 0.3);
        color: #1e40af;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
        font-size: 0.9em;
        border-left: 3px solid #3b82f6;
    }
    
    /* Document cards */
    .doc-card {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(236, 72, 153, 0.1) 100%);
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #8b5cf6 0%, #ec4899 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'response_times' not in st.session_state:
    st.session_state.response_times = []
if 'cache' not in st.session_state:
    st.session_state.cache = {}
if 'total_chunks' not in st.session_state:
    st.session_state.total_chunks = 0
if 'llm_loaded' not in st.session_state:
    st.session_state.llm_loaded = False

def format_docs(docs):
    """Format retrieved documents for context"""
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def load_local_llm(model_name):
    """Load local LLM model - cached to prevent reloading"""
    try:
        if "flan-t5" in model_name:
            # Seq2Seq models like FLAN-T5
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float32
            )
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.3,
                do_sample=True
            )
        else:
            # Causal LM models
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                repetition_penalty=1.1
            )
        
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Model Configuration")
    
    # Model selection
    embedding_model = st.selectbox(
        "Embedding Model",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-MiniLM-L3-v2",
            "sentence-transformers/all-mpnet-base-v2"
        ],
        help="Smaller models = faster processing"
    )
    
    # LLM selection - only small local models
    llm_model = st.selectbox(
        "Language Model (Local)",
        [
            "google/flan-t5-small",
            "google/flan-t5-base",
            "distilgpt2",
            "gpt2"
        ],
        help="Small models that run locally without API"
    )
    
    st.info("💡 Using fully local models - No API token needed!")
    
    # Load model button
    if st.button("📥 Load Model", use_container_width=True):
        with st.spinner(f"Loading {llm_model}... This may take a minute..."):
            llm = load_local_llm(llm_model)
            if llm:
                st.session_state.llm = llm
                st.session_state.llm_loaded = True
                st.success("✅ Model loaded successfully!")
            else:
                st.error("❌ Failed to load model")
    
    if st.session_state.llm_loaded:
        st.success("✅ Model Ready")
    else:
        st.warning("⚠️ Please load the model first")
    
    st.markdown("---")
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        chunk_size = st.slider("Chunk Size", 200, 2000, 800)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 150)
        k_documents = st.slider("Documents to Retrieve", 1, 10, 3)
    
    st.markdown("---")
    
    # Statistics
    st.markdown("### 📊 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", len(st.session_state.documents))
        st.metric("Total Chunks", st.session_state.total_chunks)
    with col2:
        st.metric("Messages", len(st.session_state.messages))
        avg_time = sum(st.session_state.response_times) / len(st.session_state.response_times) if st.session_state.response_times else 0
        st.metric("Avg Response", f"{avg_time:.2f}s")
    
    st.markdown("---")
    
    # Actions
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.cache = {}
        st.rerun()
    
    if st.button("📥 Export Chat", use_container_width=True):
        if st.session_state.messages:
            chat_export = "\n\n".join([
                f"[{msg['timestamp']}] {msg['role'].upper()}: {msg['content']}"
                for msg in st.session_state.messages
            ])
            st.download_button(
                "Download Chat Log",
                chat_export,
                file_name=f"chat_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

# Main content
st.markdown("# 🤖 RAG Document Q&A System")
st.markdown("### 🏠 100% Offline - No API Required!")

# Document upload section
st.markdown("## 📁 Document Upload")
uploaded_files = st.file_uploader(
    "Upload your documents (PDF, TXT, DOCX)",
    type=['pdf', 'txt', 'docx'],
    accept_multiple_files=True,
    help="Upload multiple documents to build your knowledge base"
)

# Process documents
if uploaded_files and st.button("🚀 Process Documents"):
    if not st.session_state.llm_loaded:
        st.error("❌ Please load the LLM model in the sidebar first!")
    else:
        with st.spinner("Processing documents..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Initialize embeddings
                status_text.text("Loading embedding model...")
                progress_bar.progress(10)
                
                if st.session_state.embeddings is None:
                    st.session_state.embeddings = HuggingFaceEmbeddings(
                        model_name=embedding_model,
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={'normalize_embeddings': True}
                    )
                
                # Process each file
                all_documents = []
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    progress_bar.progress(20 + (idx * 40 // len(uploaded_files)))
                    
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Load document based on type
                    try:
                        if uploaded_file.name.endswith('.pdf'):
                            loader = PyPDFLoader(tmp_path)
                        elif uploaded_file.name.endswith('.txt'):
                            loader = TextLoader(tmp_path)
                        elif uploaded_file.name.endswith('.docx'):
                            loader = Docx2txtLoader(tmp_path)
                        
                        documents = loader.load()
                        
                        # Store document info
                        st.session_state.documents.append({
                            'name': uploaded_file.name,
                            'size': len(uploaded_file.getvalue()),
                            'pages': len(documents),
                            'uploaded_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                        all_documents.extend(documents)
                    finally:
                        os.unlink(tmp_path)
                
                # Split documents
                status_text.text("Splitting documents into chunks...")
                progress_bar.progress(60)
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len
                )
                splits = text_splitter.split_documents(all_documents)
                st.session_state.total_chunks = len(splits)
                
                # Create vector store
                status_text.text("Creating vector embeddings...")
                progress_bar.progress(80)
                
                if st.session_state.vectorstore is None:
                    st.session_state.vectorstore = Chroma.from_documents(
                        documents=splits,
                        embedding=st.session_state.embeddings,
                        persist_directory="./chroma_db"
                    )
                else:
                    st.session_state.vectorstore.add_documents(splits)
                
                # Create RAG chain
                status_text.text("Setting up Q&A system...")
                progress_bar.progress(90)
                
                # Create prompt template
                template = """Answer the question based on the following context. If you cannot answer based on the context, say "I don't have enough information to answer that."

Context: {context}

Question: {question}

Answer:"""
                
                prompt = PromptTemplate.from_template(template)
                
                # Create retriever
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": k_documents}
                )
                
                # Create RAG chain
                st.session_state.rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | st.session_state.llm
                    | StrOutputParser()
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")
                time.sleep(1)
                st.success(f"Successfully processed {len(uploaded_files)} document(s) with {len(splits)} chunks!")
                st.balloons()
                
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# Display uploaded documents
if st.session_state.documents:
    st.markdown("## 📚 Loaded Documents")
    for doc in st.session_state.documents:
        st.markdown(f"""
        <div class="doc-card">
            <strong>📄 {doc['name']}</strong><br>
            Size: {doc['size'] / 1024:.2f} KB | Pages/Sections: {doc['pages']} | Uploaded: {doc['uploaded_at']}
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Chat interface
st.markdown("## 💬 Chat with Your Documents")

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        if message['role'] == 'system':
            st.markdown(f"""
            <div class="system-message">
                ℹ️ {message['content']}
            </div>
            """, unsafe_allow_html=True)
        elif message['role'] == 'user':
            st.markdown(f"""
            <div class="user-message">
                <strong>You:</strong> {message['content']}<br>
                <small style="opacity: 0.8;">{message['timestamp']}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            response_time = message.get('response_time', '')
            st.markdown(f"""
            <div class="assistant-message">
                <strong>🤖 Assistant:</strong> {message['content']}<br>
                <small style="opacity: 0.7;">{message['timestamp']} {response_time}</small>
            </div>
            """, unsafe_allow_html=True)

# Chat input
if st.session_state.rag_chain is not None:
    user_query = st.text_input(
        "Ask a question about your documents:",
        placeholder="What would you like to know?",
        key="user_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        send_button = st.button("🚀 Send", use_container_width=True)
    with col2:
        use_cache = st.checkbox("Use Cache", value=True)
    
    if send_button and user_query:
        # Add user message
        st.session_state.messages.append({
            'role': 'user',
            'content': user_query,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
        # Check cache
        cache_key = hashlib.md5(user_query.lower().encode()).hexdigest()
        
        if use_cache and cache_key in st.session_state.cache:
            # Use cached response
            cached_response = st.session_state.cache[cache_key]
            st.session_state.messages.append({
                'role': 'assistant',
                'content': f"⚡ (Cached) {cached_response}",
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'response_time': '(0.01s)'
            })
            st.rerun()
        else:
            # Generate new response
            with st.spinner("🤔 Thinking..."):
                start_time = time.time()
                
                try:
                    answer = st.session_state.rag_chain.invoke(user_query)
                    
                    # Clean up the answer (remove repetitions and extra text)
                    if isinstance(answer, str):
                        answer = answer.strip()
                        # For text-generation models, extract first complete response
                        if "Answer:" in answer:
                            answer = answer.split("Answer:")[-1].strip()
                    
                    response_time = time.time() - start_time
                    st.session_state.response_times.append(response_time)
                    
                    # Cache the response
                    st.session_state.cache[cache_key] = answer
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': answer,
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'response_time': f"({response_time:.2f}s)"
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
else:
    if not st.session_state.llm_loaded:
        st.info("👈 Please load the LLM model from the sidebar first!")
    else:
        st.info("👆 Please upload and process documents to start chatting!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; font-size: 0.9em;'>
    <p>🏠 100% Offline | 🔒 Privacy First | 🚀 Optimized for Speed | 💾 Smart Caching</p>
    <p>💡 <strong>Pro Tip:</strong> Smaller models are faster but less accurate. FLAN-T5 models work best!</p>
</div>
""", unsafe_allow_html=True)