import streamlit as st
from openai import OpenAI
from google import genai
from core.config import config


def run_llm(provider, model_name, messages, max_tokens=500):

    if provider == "OpenAI":
        client = OpenAI(api_key=config.OPENAI_API_KEY)

    else:
        client = genai.Client(api_key=config.GOOGLE_API_KEY)

    if provider == "Google":
        return client.models.generate_content(
            model=model_name,
            contents=[message["content"] for message in messages],
        ).text
    else:
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
            reasoning_effort="minimal"
        ).choices[0].message.content

    with st.sidebar:
        st.title("LLM API")
        provider = st.selectbox("Select LLM Provider", ["OpenAI", "Google"])
        model_name = st.text_input("Enter Model Name", "gpt-6-luna")
        messages = st.text_area("Enter Messages (JSON format)", '[{"role": "user", "content": "Hello!"}]')
        max_tokens = st.number_input("Max Tokens", min_value=1, value=500)
