"""
Web API for the movie ticket agent.

Run:   uvicorn server:app --host 0.0.0.0 --port 8000
Open:  http://127.0.0.1:8000
"""
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent import final_graph

log = logging.getLogger("cinema")
BASE = Path(__file__).parent

app = FastAPI(title="Movie Ticket Agent", docs_url=None, redoc_url=None)

# ---- tiny in-memory rate limiter (use slowapi/Redis if you run several servers) ----
LIMIT, WINDOW = 20, 60  # 20 requests per 60 seconds per IP
_hits: dict[str, deque] = defaultdict(deque)


def check_rate(request: Request):
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0].strip()
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > WINDOW:
        q.popleft()
    if len(q) >= LIMIT:
        raise HTTPException(429, "Too many requests. Please wait a minute and try again.")
    q.append(now)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=500)


class ConfirmIn(BaseModel):
    thread_id: str = Field(min_length=8, max_length=64)
    answer: str = Field(pattern="^(yes|no)$")


def clean(obj):
    """Make state JSON-safe (datetime, Decimal, etc. become strings)."""
    return json.loads(json.dumps(obj, default=str))


def pack(res: dict, thread_id: str) -> dict:
    # Graph paused at interrupt() -> ask the customer to confirm
    if res.get("__interrupt__"):
        v = res["__interrupt__"][0].value
        v = v if isinstance(v, dict) else {"message": str(v)}
        return clean({
            "thread_id": thread_id,
            "status": "confirm",
            "prompt": v.get("message", "Confirm your booking?"),
            "summary": {
                "movie": v.get("movie"),
                "customer": v.get("customer"),
                "seats": v.get("seats"),
                "total_amount": v.get("total_amount"),
            },
        })
    return clean({
        "thread_id": thread_id,
        "status": "done",
        "result": res.get("result", ""),
        "data": res.get("data", {}),
    })


def fail(e: Exception):
    log.exception("agent error")  # details go to the server log, not to the customer
    return JSONResponse(
        {"status": "error", "message": "Something went wrong on our side. Please try again."},
        status_code=500,
    )


@app.post("/api/chat")
def chat(body: ChatIn, request: Request):
    check_rate(request)
    thread_id = str(uuid.uuid4())  # new thread per request: no old state leaks between customers
    config = {"configurable": {"thread_id": thread_id}}
    try:
        res = final_graph.invoke({"user_message": body.message.strip()}, config)
        return pack(res, thread_id)
    except Exception as e:
        return fail(e)


@app.post("/api/confirm")
def confirm(body: ConfirmIn, request: Request):
    check_rate(request)
    config = {"configurable": {"thread_id": body.thread_id}}
    try:
        # Only resume a thread that is really waiting for confirmation
        if not final_graph.get_state(config).next:
            raise HTTPException(409, "This booking is no longer waiting for confirmation.")
        res = final_graph.invoke(Command(resume=body.answer), config)
        return pack(res, body.thread_id)
    except HTTPException:
        raise
    except Exception as e:
        return fail(e)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(BASE / "index.html")