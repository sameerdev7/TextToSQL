from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
from typing import Optional
import re

app = FastAPI(title="Text-to-SQL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "../models/phi35-text2sql-q4.gguf"
LLAMA_SERVER_URL = "http://localhost:8080"

SYSTEM_PROMPT = (
    "You are an expert Text-to-SQL assistant. "
    "Return ONLY executable SQL for the given question and schema. "
    "Do not include explanations, comments, or markdown. "
    "Prefer ANSI SQL; use tables/columns exactly as provided."
)

class SQLRequest(BaseModel):
    db_schema: str  # Renamed from 'schema' to avoid shadowing
    question: str
    context_length: Optional[int] = 1024
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.1

class SQLResponse(BaseModel):
    sql: str
    raw_output: str
    success: bool
    error: Optional[str] = None

def extract_sql(text: str) -> str:
    """Extract SQL from model output, removing markdown and extra text."""
    text = text.strip()
    
    # Remove markdown code blocks
    text = re.sub(r'```sql\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Remove common prefixes
    prefixes = [
        "Here's the SQL:",
        "Here is the SQL:",
        "SQL:",
        "Query:",
        "The SQL query is:",
        "Sure, based on the given schema and request, here's the SQL query that will return the desired result:",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    
    # Take only the first SQL statement (until semicolon or end)
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Stop at explanatory text
        if any(line.lower().startswith(word) for word in ['this', 'the', 'note:', 'explanation:']):
            break
        lines.append(line)
        if line.endswith(';'):
            break
    
    return ' '.join(lines).strip()

def check_llama_server() -> bool:
    """Check if llama-server is running."""
    try:
        response = requests.get(f"{LLAMA_SERVER_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def generate_with_llama_server(prompt: str, max_tokens: int, temperature: float) -> str:
    """Generate text using llama-server API."""
    
    if not check_llama_server():
        raise HTTPException(
            status_code=503,
            detail=f"llama-server is not running at {LLAMA_SERVER_URL}. Start it with: ./start_llama_server.sh"
        )
    
    try:
        response = requests.post(
            f"{LLAMA_SERVER_URL}/completion",
            json={
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
                "stop": ["\n>", "Question:", "Schema:"],  # Stop tokens
                "stream": False,
            },
            timeout=None  # No timeout
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get("content", "")
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error communicating with llama-server: {str(e)}"
        )

@app.get("/")
def root():
    return {
        "message": "Text-to-SQL API",
        "model": MODEL_PATH,
        "llama_server": LLAMA_SERVER_URL,
        "status": "running"
    }

@app.get("/health")
def health_check():
    """Check if the model and llama-server are available."""
    model_exists = os.path.exists(MODEL_PATH)
    server_running = check_llama_server()
    
    return {
        "status": "healthy" if (model_exists and server_running) else "unhealthy",
        "model_available": model_exists,
        "llama_server_running": server_running,
        "model_path": MODEL_PATH,
        "llama_server_url": LLAMA_SERVER_URL
    }

@app.post("/generate-sql", response_model=SQLResponse)
def generate_sql(request: SQLRequest):
    """Generate SQL from natural language question and schema."""
    
    # Format the prompt with system message
    prompt = f"""<|system|>
{SYSTEM_PROMPT}
<|user|>
Schema: {request.db_schema}
Question: {request.question}
<|assistant|>
"""
    
    try:
        # Run inference via llama-server
        raw_output = generate_with_llama_server(
            prompt,
            request.max_tokens,
            request.temperature
        )
        
        # Extract clean SQL
        sql = extract_sql(raw_output)
        
        if not sql:
            return SQLResponse(
                sql="",
                raw_output=raw_output,
                success=False,
                error="No SQL generated from model output"
            )
        
        return SQLResponse(
            sql=sql,
            raw_output=raw_output,
            success=True
        )
        
    except Exception as e:
        return SQLResponse(
            sql="",
            raw_output="",
            success=False,
            error=str(e)
        )

@app.post("/generate-sql-batch")
def generate_sql_batch(requests: list[SQLRequest]):
    """Generate SQL for multiple questions."""
    results = []
    for req in requests:
        result = generate_sql(req)
        results.append(result)
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
