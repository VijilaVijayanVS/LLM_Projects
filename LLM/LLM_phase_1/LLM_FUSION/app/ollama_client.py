import subprocess

def query_ollama(prompt: str, model: str = "mistral") -> str:
    result = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

def get_chat_response(message: str) -> str:
    prompt = f"User: {message}\nAssistant:"
    return query_ollama(prompt)

def get_summary(document_text: str) -> str:
    prompt = f"Summarize this document:\n\n{document_text}\n\nSummary:"
    return query_ollama(prompt)
