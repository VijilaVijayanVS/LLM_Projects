from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from ollama_client import get_chat_response, get_summary
from doc_parser import extract_text

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    reply = get_chat_response(request.message)
    return {"response": reply}

@app.post("/summarize")
def summarize(file: UploadFile = File(...)):
    content = extract_text(file)
    summary = get_summary(content)
    return {"summary": summary}