import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

ROUTER_WEIGHTS_PATH = BASE_DIR / "app" / "models" / "router_weights.pt"
LABEL_MAP_PATH = BASE_DIR / "app" / "models" / "label_map.json"
FAISS_INDEX_DIR = BASE_DIR / "data" / "faiss_index"
DOCS_DIR = BASE_DIR / "data" / "documents"

MAX_HOPS = 3
RETRIEVAL_TOP_K = 4
