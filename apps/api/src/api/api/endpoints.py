import logging
from fastapi import APIRouter, Request

from api.api.models import RAGRequest, RAGResponse
from qdrant_client import QdrantClient

from api.agents.retireval_generation import rag_pipline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

rag_router = APIRouter()

qdrant_client = QdrantClient(url="http://qdrant:6333")

@rag_router.post("/")
def chat(
    request: Request,
    payload: RAGRequest
) -> RAGResponse:
    result = rag_pipline(payload.query, qdrant_client)
    return RAGResponse(answer=result["answer"], products=result["products"])


api_router = APIRouter()
api_router.include_router(rag_router,prefix="/rag",tags=["rag"])