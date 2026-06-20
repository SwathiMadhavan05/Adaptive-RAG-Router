"""
FastAPI app exposing the adaptive RAG router as a REST API.

Run with:
    uvicorn app.main:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive API docs.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.vector_store import VectorStore
from app.orchestrator import Orchestrator
from app.config import FAISS_INDEX_DIR

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading vector store and router classifier...")
    store = VectorStore()
    index_path = str(FAISS_INDEX_DIR / "index")
    try:
        store.load(index_path)
    except FileNotFoundError:
        raise RuntimeError(
            "FAISS index not found. Run `python scripts/build_index.py` first."
        )

    state["orchestrator"] = Orchestrator(vector_store=store)
    print("Ready.")
    yield
    state.clear()


app = FastAPI(
    title="Adaptive RAG Router",
    description="Routes queries to the cheapest sufficient strategy: parametric memory, simple RAG, multi-hop RAG, or web search.",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
async def root():
    return FileResponse("frontend/landing.html")

@app.get("/dashboard")
async def dashboard():
    return FileResponse("frontend/index.html")



class QueryRequest(BaseModel):
    query: str
    force_route: str | None = None  # for eval/baseline comparisons only


class QueryResponse(BaseModel):
    query: str
    answer: str
    route: str
    router_confidence: float
    router_probs: dict
    hops: int
    latency_seconds: float
    cost_units: int
    chunks_retrieved: list


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    orchestrator: Orchestrator = state["orchestrator"]
    result = orchestrator.handle_query(request.query, force_route=request.force_route)
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}
