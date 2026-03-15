#!/usr/bin/env python3
"""
wheelson-explorer · middleware · main.py
════════════════════════════════════════
Autonomous exploration loop powered by Google Gemini / Ollama.
Serves a live browser dashboard over Server-Sent Events.

Usage
    python main.py --personality benson
    python main.py --personality sir_david
    python main.py --personality klaus
    python main.py --personality zog7

Dashboard
    http://localhost:8000
"""

import argparse
import asyncio
import logging
import sys

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from state import state
from config import PERSONALITIES, GEMINI_API_KEY, GEMINI_MODEL, OLLAMA_MODEL, PORT
import config as _config_module
from loop import explorer_loop
from routes import dashboard, sse_stream, snapshot, health
from remote import (
    JoinQueueRequest,
    LeaveQueueRequest,
    RemoteControlRequest,
    PersonaRequest,
    ChatRequest,
    queue_join,
    queue_leave,
    remote_control,
    switch_persona,
    chat_endpoint,
    queue_status,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("wheelson")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Wheelson Explorer", docs_url=None, redoc_url=None)


@app.on_event("startup")
async def _startup() -> None:
    asyncio.create_task(explorer_loop())


@app.get("/", response_class=HTMLResponse)
async def _dashboard() -> str:
    return await dashboard()


@app.get("/stream")
async def _sse_stream():
    return await sse_stream()


@app.post("/queue/join")
async def _queue_join(req: JoinQueueRequest):
    return await queue_join(req)


@app.post("/queue/leave")
async def _queue_leave(req: LeaveQueueRequest):
    return await queue_leave(req)


@app.post("/control")
async def _remote_control(req: RemoteControlRequest):
    return await remote_control(req)


@app.post("/persona")
async def _switch_persona(req: PersonaRequest):
    return await switch_persona(req)


@app.get("/queue/status")
async def _queue_status():
    return await queue_status()


@app.post("/chat")
async def _chat_endpoint(req: ChatRequest):
    return await chat_endpoint(req)


@app.get("/snapshot")
async def _snapshot():
    return await snapshot()


@app.get("/health")
async def _health():
    return await health()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(
        description="Wheelson Explorer \u2014 AI-powered autonomous robot middleware"
    )
    parser.add_argument(
        "--personality",
        choices=list(PERSONALITIES.keys()),
        default="benson",
        metavar="NAME",
        help=f"Active personality. Choices: {', '.join(PERSONALITIES.keys())}",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "ollama"],
        default="gemini",
        help="VLM provider: 'gemini' (cloud) or 'ollama' (local). Default: gemini",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="Model name override. Defaults: gemini=gemini-2.5-flash, ollama=llava",
    )
    parser.add_argument("--port", type=int, default=PORT, help="Dashboard port (default: 8000)")
    args = parser.parse_args()

    _config_module.PROVIDER = args.provider
    if args.model:
        _config_module.ACTIVE_MODEL = args.model
    elif _config_module.PROVIDER == "gemini":
        _config_module.ACTIVE_MODEL = GEMINI_MODEL
    else:
        _config_module.ACTIVE_MODEL = OLLAMA_MODEL

    if _config_module.PROVIDER == "gemini" and not GEMINI_API_KEY:
        log.error("\u274c GEMINI_API_KEY is not set. Add it to your .env file and retry.")
        sys.exit(1)

    state.personality_key = args.personality
    p = PERSONALITIES[args.personality]

    log.info("\u2550" * 55)
    log.info("  Wheelson Autonomous Explorer")
    log.info("  Personality : %s  %s", p["emoji"], p["name"])
    log.info("  Provider    : %s  (model: %s)", _config_module.PROVIDER.upper(), _config_module.ACTIVE_MODEL)
    log.info("  Move bias   : %s", p["move_bias"])
    log.info("  Wheelson IP : %s", _config_module.WHEELSON_IP)
    log.info("  Dashboard   : http://localhost:%d", args.port)
    log.info("\u2550" * 55)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
