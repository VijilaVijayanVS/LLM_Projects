import streamlit as st
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
import librosa
import numpy as np
from datetime import datetime
import json
import re
from io import BytesIO
import soundfile as sf

# Page configuration
st.set_page_config(
    page_title="Minute Master",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for pastel colors and enhanced UI
st.markdown("""
<style>
    /* Main background with pastel gradient */
    .stApp {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 25%, #e0c3fc 50%, #8ec5fc 75%, #c2e9fb 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffecd2 0%, #fcb69f 100%);
    }
    
    /* Cards and containers */
    .stMarkdown, .stAlert {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Headers */
    h1 {
        color: #6B46C1;
        text-align: center;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h2 {
        color: #805AD5;
        font-weight: 700;
    }
    
    h3 {
        color: #9F7AEA;
        font-weight: 600;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 15px;
        padding: 20px;
        border: 2px dashed #9F7AEA;
    }
    
    /* Success/Info boxes */
    .stSuccess {
        background-color: #D4EDDA;
        border-left: 5px solid #28A745;
    }
    
    .stInfo {
        background-color: #D1ECF1;
        border-left: 5px solid #17A2B8;
    }
    
    .stWarning {
        background-color: #FFF3CD;
        border-left: 5px solid #FFC107;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'action_items' not in st.session_state:
    st.session_state.action_items = None
if 'key_points' not in st.session_state:
    st.session_state.key_points = None
if 'participants' not in st.session_state:
    st.session_state.participants = None

# Cache model loading
@st.cache_resource
def load_whisper_model(model_size="base"):
    """Load Whisper model from Hugging Face"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        model_id = f"openai/whisper-{model_size}"
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(device)
        
        processor = AutoProcessor.from_pretrained(model_id)
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )
        
        return pipe
    except Exception as e:
        st.error(f"Error loading Whisper model: {str(e)}")
        return None

@st.cache_resource
def load_summarization_model():
    """Load summarization model from Hugging Face"""
    try:
        model_name = "facebook/bart-large-cnn"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        summarizer = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        return summarizer
    except Exception as e:
        st.error(f"Error loading summarization model: {str(e)}")
        return None

@st.cache_data
def process_audio(audio_bytes, sample_rate=16000):
    """Process audio file and convert to required format"""
    try:
        # Load audio using soundfile
        audio_data, sr = sf.read(BytesIO(audio_bytes))
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample to 16kHz if needed
        if sr != sample_rate:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=sample_rate)
        
        return audio_data.astype(np.float32)
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

def transcribe_audio(audio_data, whisper_pipe):
    """Transcribe audio using Whisper"""
    try:
        result = whisper_pipe(audio_data)
        return result["text"]
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return None

def generate_summary(text, summarizer):
    """Generate meeting summary"""
    try:
        # Split text into chunks if too long
        max_chunk_length = 1024
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_length += len(word) + 1
            if current_length > max_chunk_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # Summarize each chunk
        summaries = []
        for chunk in chunks:
            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        
        return " ".join(summaries)
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

def extract_action_items(text):
    """Extract action items using rule-based approach and patterns"""
    action_items = []
    
    # Action verbs and patterns
    action_patterns = [
        r"(?:will|should|must|need to|have to|going to)\s+([^.!?]+)",
        r"(?:action item|task|todo|to-do):\s*([^.!?]+)",
        r"(?:please|can you|could you)\s+([^.!?]+)",
        r"([A-Z][a-z]+)\s+(?:will|should|must|needs? to)\s+([^.!?]+)",
    ]
    
    sentences = re.split(r'[.!?]\s+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        for pattern in action_patterns:
            matches = re.finditer(pattern, sentence, re.IGNORECASE)
            for match in matches:
                action = match.group(0).strip()
                if len(action) > 15 and len(action) < 200:
                    # Extract potential assignee
                    assignee_match = re.search(r'\b([A-Z][a-z]+)\b', action)
                    assignee = assignee_match.group(1) if assignee_match else "Unassigned"
                    
                    # Determine priority based on keywords
                    priority = "Medium"
                    if any(word in action.lower() for word in ["urgent", "asap", "immediately", "critical"]):
                        priority = "High"
                    elif any(word in action.lower() for word in ["later", "eventually", "consider"]):
                        priority = "Low"
                    
                    action_items.append({
                        "task": action,
                        "assignee": assignee,
                        "priority": priority,
                        "deadline": "TBD"
                    })
    
    # Remove duplicates
    unique_actions = []
    seen = set()
    for item in action_items:
        task_lower = item['task'].lower()
        if task_lower not in seen:
            seen.add(task_lower)
            unique_actions.append(item)
    
    return unique_actions[:10]  # Return top 10

def extract_key_points(text):
    """Extract key points from text"""
    sentences = re.split(r'[.!?]\s+', text)
    
    # Score sentences based on important keywords
    important_keywords = [
        'important', 'key', 'significant', 'critical', 'essential', 'main',
        'decided', 'agreed', 'conclusion', 'result', 'outcome', 'goal',
        'objective', 'priority', 'focus', 'issue', 'problem', 'solution'
    ]
    
    scored_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20:
            score = sum(1 for keyword in important_keywords if keyword in sentence.lower())
            if score > 0:
                scored_sentences.append((score, sentence))
    
    # Sort by score and return top points
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    return [sent for score, sent in scored_sentences[:7]]

def extract_participants(text):
    """Extract participant names from text"""
    # Look for names (capitalized words)
    name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
    names = re.findall(name_pattern, text)
    
    # Filter out common words
    common_words = {'Speaker', 'Meeting', 'Today', 'Tomorrow', 'Thanks', 'Please', 'Good', 'Morning', 'Afternoon'}
    participants = [name for name in set(names) if name not in common_words]
    
    return participants[:8]  # Return max 8 participants

def create_download_content():
    """Create downloadable content"""
    content = f"""
MEETING NOTES AND ACTION ITEMS
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

PARTICIPANTS
{', '.join(st.session_state.participants) if st.session_state.participants else 'N/A'}

{'='*60}
MEETING SUMMARY
{st.session_state.summary}

{'='*60}
KEY POINTS
"""
    for i, point in enumerate(st.session_state.key_points or [], 1):
        content += f"\n{i}. {point}"
    
    content += f"\n\n{'='*60}\nACTION ITEMS\n"
    for i, item in enumerate(st.session_state.action_items or [], 1):
        content += f"\n{i}. [{item['priority']}] {item['task']}"
        content += f"\n   Assignee: {item['assignee']}"
        content += f"\n   Deadline: {item['deadline']}\n"
    
    content += f"\n{'='*60}\nFULL TRANSCRIPT\n{st.session_state.transcript}"
    
    return content

# Main UI
st.title("🎤 Minute Master")
st.markdown("### Transform your meeting audio into structured notes with AI")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    model_size = st.selectbox(
        "Whisper Model Size",
        ["tiny", "base", "small", "medium"],
        index=1,
        help="Smaller models are faster but less accurate. Base is recommended for balance."
    )
    
    st.info(f"**Device:** {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    
    st.markdown("---")
    
    st.header("📊 Model Info")
    st.markdown("""
    **Whisper Model:**
    - Speech-to-text transcription
    - Supports multiple languages
    
    **BART Model:**
    - Text summarization
    - Action item extraction
    """)
    
    st.markdown("---")
    
    st.header("🎯 Features")
    st.markdown("""
    ✅ Audio transcription  
    ✅ Meeting summary  
    ✅ Action items extraction  
    ✅ Key points identification  
    ✅ Participant detection  
    ✅ Downloadable notes  
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📁 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        help="Upload your meeting recording"
    )

with col2:
    st.header("ℹ️ Instructions")
    st.markdown("""
    1. Upload audio file
    2. Click 'Process Audio'
    3. Wait for analysis
    4. Review & download results
    """)

if uploaded_file is not None:
    st.success(f"✅ File uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024 / 1024:.2f} MB)")
    
    # Audio player
    st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        process_button = st.button("🚀 Process Audio", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear Results", use_container_width=True)
    
    if clear_button:
        st.session_state.transcript = None
        st.session_state.summary = None
        st.session_state.action_items = None
        st.session_state.key_points = None
        st.session_state.participants = None
        st.rerun()
    
    if process_button:
        start_time = datetime.now()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Load models
            status_text.text("📥 Loading AI models...")
            progress_bar.progress(10)
            
            whisper_pipe = load_whisper_model(model_size)
            summarizer = load_summarization_model()
            
            if not whisper_pipe or not summarizer:
                st.error("Failed to load models. Please try again.")
                st.stop()
            
            progress_bar.progress(25)
            
            # Step 2: Process audio
            status_text.text("🎵 Processing audio file...")
            audio_bytes = uploaded_file.read()
            audio_data = process_audio(audio_bytes)
            
            if audio_data is None:
                st.error("Failed to process audio. Please check the file format.")
                st.stop()
            
            progress_bar.progress(40)
            
            # Step 3: Transcribe
            status_text.text("🎤 Transcribing audio... (this may take a moment)")
            transcript = transcribe_audio(audio_data, whisper_pipe)
            
            if not transcript:
                st.error("Failed to transcribe audio. Please try again.")
                st.stop()
            
            st.session_state.transcript = transcript
            progress_bar.progress(60)
            
            # Step 4: Generate summary
            status_text.text("📝 Generating meeting summary...")
            summary = generate_summary(transcript, summarizer)
            st.session_state.summary = summary
            progress_bar.progress(75)
            
            # Step 5: Extract action items
            status_text.text("✅ Extracting action items...")
            action_items = extract_action_items(transcript)
            st.session_state.action_items = action_items
            progress_bar.progress(85)
            
            # Step 6: Extract key points and participants
            status_text.text("🔍 Identifying key points and participants...")
            key_points = extract_key_points(transcript)
            participants = extract_participants(transcript)
            st.session_state.key_points = key_points
            st.session_state.participants = participants
            progress_bar.progress(100)
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            status_text.empty()
            progress_bar.empty()
            
            st.success(f"✅ Processing complete in {processing_time:.2f} seconds!")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            progress_bar.empty()
            status_text.empty()

# Display results
if st.session_state.transcript:
    st.markdown("---")
    st.header("📊 Analysis Results")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Word Count",
            len(st.session_state.transcript.split()),
            help="Total words in transcript"
        )
    
    with col2:
        st.metric(
            "Action Items",
            len(st.session_state.action_items or []),
            help="Extracted action items"
        )
    
    with col3:
        st.metric(
            "Key Points",
            len(st.session_state.key_points or []),
            help="Important discussion points"
        )
    
    with col4:
        st.metric(
            "Participants",
            len(st.session_state.participants or []),
            help="Detected speakers"
        )
    
    # Download button
    st.download_button(
        label="📥 Download Complete Notes",
        data=create_download_content(),
        file_name=f"meeting_notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True
    )
    
    # Tabs for organized display
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Summary", "✅ Action Items", "🎯 Key Points", "👥 Participants", "📄 Full Transcript"
    ])
    
    with tab1:
        st.subheader("Meeting Summary")
        st.info(st.session_state.summary)
    
    with tab2:
        st.subheader("Action Items")
        if st.session_state.action_items:
            for i, item in enumerate(st.session_state.action_items, 1):
                priority_color = {
                    "High": "🔴",
                    "Medium": "🟡",
                    "Low": "🟢"
                }
                
                with st.expander(f"{priority_color[item['priority']]} {i}. {item['task'][:80]}..."):
                    st.markdown(f"**Full Task:** {item['task']}")
                    st.markdown(f"**Assignee:** {item['assignee']}")
                    st.markdown(f"**Priority:** {item['priority']}")
                    st.markdown(f"**Deadline:** {item['deadline']}")
        else:
            st.warning("No action items found in the transcript.")
    
    with tab3:
        st.subheader("Key Discussion Points")
        if st.session_state.key_points:
            for i, point in enumerate(st.session_state.key_points, 1):
                st.markdown(f"**{i}.** {point}")
        else:
            st.warning("No key points identified.")
    
    with tab4:
        st.subheader("Meeting Participants")
        if st.session_state.participants:
            cols = st.columns(4)
            for i, participant in enumerate(st.session_state.participants):
                with cols[i % 4]:
                    st.markdown(f"👤 **{participant}**")
        else:
            st.info("No specific participants detected in the transcript.")
    
    with tab5:
        st.subheader("Complete Transcript")
        st.text_area(
            "Full transcript",
            st.session_state.transcript,
            height=400,
            label_visibility="collapsed"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🚀 Built with Streamlit | 🤗 Powered by Hugging Face Models</p>
    <p>Whisper (Speech Recognition) + BART (Summarization)</p>
</div>
""", unsafe_allow_html=True)