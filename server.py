"""Run with uvicorn server:app --host 127.0.0.1 --port 8000."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.routes import router
from core.config_loader import load_config
from core.jobs import recover_interrupted_jobs


@asynccontextmanager
async def lifespan(app):
    load_config()
    recover_interrupted_jobs()
    yield


app = FastAPI(title='RAG Ablation Studio API', version='0.1.0', lifespan=lifespan)
app.include_router(router)
