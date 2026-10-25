import requests
import streamlit as st

from chatbot_ui.core.config import config

PROVIDER_MODELS = {
    "OpenAI": ["gpt-4o-mini", "gpt-5-mini"],
    "Groq": ["llama-3.3-70b-versatile"],
    "Google": ["gemini-2.5-flash"],
}

st.title("AI Engineering Bootcamp Chatbot")

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("Provider", list(PROVIDER_MODELS))
    model_name = st.selectbox("Model", PROVIDER_MODELS[provider])

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = requests.post(
                f"{config.API_URL.rstrip('/')}/chat",
                json={
                    "provider": provider,
                    "model_name": model_name,
                    "messages": st.session_state.messages,
                },
                timeout=120,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            try:
                detail = exc.response.json().get("detail", str(exc))
            except (AttributeError, ValueError):
                detail = str(exc)
            st.error(f"API error: {detail}")
        except requests.RequestException as exc:
            st.error(f"Could not reach the API service: {exc}")
        else:
            try:
                answer = response.json()["message"]
            except (KeyError, TypeError, ValueError) as exc:
                st.error(f"The API returned an invalid response: {exc}")
            else:
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
