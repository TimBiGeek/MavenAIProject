from pydantic import BaseModel, Field



class RAGRequest(BaseModel):
    query: str



class RAGUsedContext(BaseModel):
    id: str
    image_url: str | None = None
    price: float | None = None
    description: str

class RAGResponse(BaseModel):
    answer: str
    used_context: list[RAGUsedContext]