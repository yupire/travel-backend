import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 配置日志：INFO 级别，输出到控制台（uvicorn 会自动捕获）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# Load .env on startup so QWEATHER_* and DEEPSEEK_API_KEY are visible to os.getenv().
# python-dotenv is shipped transitively by pydantic/uvicorn; if unavailable we
# fall back to a no-op so the app still starts.
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(filename=".env", usecwd=True), override=False)
except Exception:  # pragma: no cover - dotenv missing is non-fatal
    pass

# ChatOpenAI / openai SDK will fall back to OPENAI_API_KEY if no api_key
# is passed in. Mirror the LLM-provider key into OPENAI_API_KEY so the
# client construction in agent/planner.py does not need to be touched.
_LLM_KEY = "DEEPSEEK_API_KEY"
import os as _os
_ds = _os.getenv(_LLM_KEY)
if _ds and not _os.getenv("OPENAI_API_KEY"):
    _os.environ["OPENAI_API_KEY"] = _ds

from router import router

app = FastAPI(title="AI Travel Agent", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3311", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
