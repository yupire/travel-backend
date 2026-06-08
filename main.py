from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from router import router

# Load .env on startup so QWEATHER_* and * are visible to os.getenv().
# python-dotenv is shipped transitively by pydantic/uvicorn; if unavailable we
# fall back to a no-op so the app still starts.
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(filename=".env", usecwd=True), override=False)
except Exception:  # pragma: no cover - dotenv missing is non-fatal
    pass

app = FastAPI(title="AI Travel Agent", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3311", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
