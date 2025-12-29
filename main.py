from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str

SYSTEM_PROMPT = """
You are the official AI assistant of Awasthi Digital.

Awasthi Digital is a professional digital marketing agency and training institute.

Services:
- Digital Marketing Training
- Meta Ads (Facebook & Instagram Ads)
- Google Ads
- SEO
- Website Development
- AI Marketing & Automation
- Social Media Marketing
- 1-on-1 Consulting

Rules:
- Answer only digital marketing & Awasthi Digital related queries
- Be polite, professional, and helpful
- If user asks for website or contact, share:
  https://awasthidigital.com
"""

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=200,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": req.message}
            ]
        )
        return {"reply": response.choices[0].message.content}
    except RateLimitError:
        raise HTTPException(status_code=429, detail="AI is busy. Try again shortly.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
