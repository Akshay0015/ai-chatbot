from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI, RateLimitError
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# ✅ ADD CORS (THIS FIXES YOUR ISSUE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all (safe for local dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI setup
api_key = os.getenv("OPENAI_API_KEY")
print("API KEY FOUND:", api_key is not None)

client = OpenAI(api_key=api_key)

class ChatRequest(BaseModel):
    message: str

SYSTEM_PROMPT = """
You are a helpful, polite AI assistant.
Answer clearly and briefly.
"""

@app.get("/")
def home():
    return {"status": "AI Chatbot backend is running"}

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=150,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": req.message}
            ]
        )
        return {"reply": response.choices[0].message.content}

    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="AI is busy right now. Please try again in a few minutes."
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
