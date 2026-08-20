from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .agent import LiveToolCallingAgent, OfflineToolAgent
from .knowledge import KnowledgeBase
from .retrieval import KeywordRetriever, SemanticRetriever


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = (PROJECT_ROOT / "templates" / "index.html").read_text(encoding="utf-8")

kb = KnowledgeBase()
keyword = KeywordRetriever(kb)
semantic = SemanticRetriever(kb, keyword)
offline_agent = OfflineToolAgent(kb, semantic)
live_agent = LiveToolCallingAgent(kb)

app = FastAPI(
    title="On-call Agent Lab",
    description="用合成运行手册比较关键词检索、轻量语义检索和工具调用 Agent。",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    mode: str = Field(default="offline", pattern="^(offline|live)$")


@app.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    configured = "true" if live_agent.configured else "false"
    return HTMLResponse(INDEX_HTML.replace("__LIVE_CONFIGURED__", configured))


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "runbooks": len(kb.documents), "live_configured": live_agent.configured}


@app.get("/v1/search")
async def keyword_search(q: str = Query(min_length=1, max_length=200)) -> dict:
    return {"query": q, "method": "keyword", "results": [item.to_dict() for item in keyword.search(q)]}


@app.get("/v2/search")
async def semantic_search(q: str = Query(min_length=1, max_length=200)) -> dict:
    return {
        "query": q,
        "method": "semantic-concept-vector",
        "detected_concepts": [kb.concept_label(item) for item in kb.detect_concepts(q)],
        "results": [item.to_dict() for item in semantic.search(q)],
    }


@app.get("/api/compare")
async def compare(q: str = Query(min_length=1, max_length=200)) -> dict:
    return {
        "query": q,
        "keyword": [item.to_dict() for item in keyword.search(q)],
        "semantic": [item.to_dict() for item in semantic.search(q)],
    }


@app.post("/v3/chat")
async def chat(payload: ChatRequest) -> dict:
    try:
        response = (
            live_agent.run(payload.message)
            if payload.mode == "live"
            else offline_agent.run(payload.message)
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return response.to_dict()
