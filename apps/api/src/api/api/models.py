from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    query: str


class RAGProduct(BaseModel):
    parent_asin: str
    title: str
    image_url: str | None = None
    rating: float | None = None
    similarity: float | None = None


class RAGResponse(BaseModel):
    answer: str
    products: list[RAGProduct] = Field(default_factory=list)