import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from google import genai
from groq import Groq
from openai import OpenAI
from pydantic import BaseModel, Field

from api.core.config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    provider: Literal["OpenAI", "Groq", "Google"]
    model_name: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=500, ge=1, le=8192)


class ChatResponse(BaseModel):
    message: str


def run_llm(payload: ChatRequest) -> str:
    if payload.provider == "OpenAI":
        if not config.OPENAI_API_KEY:
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not configured")
        response = OpenAI(api_key=config.OPENAI_API_KEY).chat.completions.create(
            model=payload.model_name,
            messages=[message.model_dump() for message in payload.messages],
            max_completion_tokens=payload.max_tokens,
        )
        answer = response.choices[0].message.content
    elif payload.provider == "Groq":
        if not config.GROQ_API_KEY:
            raise HTTPException(status_code=400, detail="GROQ_API_KEY is not configured")
        response = Groq(api_key=config.GROQ_API_KEY).chat.completions.create(
            model=payload.model_name,
            messages=[message.model_dump() for message in payload.messages],
            max_tokens=payload.max_tokens,
        )
        answer = response.choices[0].message.content
    else:
        if not config.GOOGLE_API_KEY:
            raise HTTPException(status_code=400, detail="GOOGLE_API_KEY is not configured")
        response = genai.Client(api_key=config.GOOGLE_API_KEY).models.generate_content(
            model=payload.model_name,
            contents=[message.content for message in payload.messages],
        )
        answer = response.text

    if not answer:
        raise HTTPException(status_code=502, detail="The model returned an empty response")
    return answer


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        return ChatResponse(message=run_llm(payload))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Provider request failed")
        raise HTTPException(status_code=502, detail="The provider request failed") from exc
