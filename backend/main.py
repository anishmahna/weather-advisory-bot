import uuid
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .graph import Advisor

app = FastAPI(title="Weather Advisory Bot")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
advisor = Advisor()          # raises at startup if any SOP file is malformed


class ChatIn(BaseModel):
    message: str
    session_id: str = ""


@app.post("/chat")
def chat(body: ChatIn):
    sid = body.session_id or str(uuid.uuid4())
    return {"session_id": sid, **advisor.chat(sid, body.message)}


@app.post("/reset/{session_id}")
def reset(session_id: str):
    advisor.memory.reset(session_id)
    return {"ok": True}


@app.get("/sops")
def sops():
    return [{"id": s.id, "category": s.category, "severity": s.severity, "title": s.title,
             "situational": s.situational, "fuzzy": bool(s.fuzzy_criterion)} for s in advisor.registry.get()]


@app.get("/health")
def health():
    return {"ok": True, "sops": len(advisor.registry.get()), "sop_load_error": advisor.registry.last_error}
