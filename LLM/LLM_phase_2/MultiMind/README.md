# Create and activate virtual env
python3 -m venv .venv
source .venv/bin/activate
 .venv\Scripts\activate

# Install required packages
pip install streamlit transformers torch pillow

# Run the app
streamlit run app.py