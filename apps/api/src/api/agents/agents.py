
from google import genai
from groq import Groq
from openai import OpenAI


from api.core.config import config

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
