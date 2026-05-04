"""
FastAPI entry point for the Meta-Agent component.

Run with:
    uvicorn main:app --reload

Mounts the API router that exposes the four endpoints defined in
requirements.md §7 (FR1, FR13, FR18, FR19).
"""

import logging
import logging.config

from fastapi import FastAPI

from api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Meta-Agent API",
    description="Persistent student modelling and learning path generation component.",
    version="0.1.0",
)

app.include_router(router)

logger.info("Meta-Agent API started.")
