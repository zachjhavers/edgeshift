import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")

from routers import mlb, nhl

app = FastAPI(title="EdgeShift API")

_origins_env = os.getenv("ALLOWED_ORIGINS", "")
origins = [o.strip() for o in _origins_env.split(",") if o.strip()] or [
    "http://localhost:3000",
    "http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(mlb.router, prefix="/api/mlb", tags=["mlb"])
app.include_router(nhl.router, prefix="/api/nhl", tags=["nhl"])


@app.get("/api/health")
def health():
    return {"status": "ok"}
