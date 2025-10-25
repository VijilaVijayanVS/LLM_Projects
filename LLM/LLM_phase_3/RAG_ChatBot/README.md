# Create and activate virtual env
python3 -m venv .venv
source .venv/bin/activate

# Install required packages
pip install streamlit langchain chromadb sentence-transformers pypdf docx2txt huggingface-hub
pip install -r requirements.txt
pip install --upgrade langchain
pip install langchain-text-splitters



# Run the app
streamlit run app.py