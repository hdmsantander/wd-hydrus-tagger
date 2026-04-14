"""Tagger API — composes HTTP and WebSocket route modules."""

from fastapi import APIRouter

from backend.routes.tagger_http import router as tagger_http_router
from backend.routes.tagger_ws import router as tagger_ws_router

router = APIRouter()
router.include_router(tagger_http_router)
router.include_router(tagger_ws_router)
