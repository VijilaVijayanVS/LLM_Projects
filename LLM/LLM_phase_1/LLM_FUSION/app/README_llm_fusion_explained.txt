======================
Execution Steps
======================

pip install -r requirements.txt

cd app
uvicorn main:app --reload

cd ui
streamlit run streamlit_app.py

======================
Code Explanation
======================

File : main.py

FastAPI is used to create the API application.
UploadFile is used to handle file uploads
File(...) is used to specify a file input parameter for an endpoint.
BaseModel is used to define data models (schemas) for request validation and parsing.
get_chat_response sends a message to an LLM (like Ollama) and gets a reply.
get_summary likely sends text to the LLM to get a summarized version.
This function likely extracts readable text from uploaded files (PDFs, DOCX, etc.).
app = FastAPI() : This app object is used to define routes and start the server.