import requests
import streamlit as st

from chatbot_ui.core.config import config


def render_products(products):
    if not products:
        return

    st.markdown("**Matching products**")
    for product in products:
        with st.container(border=True):
            image_column, details_column = st.columns([1, 3])
            with image_column:
                if product.get("image_url"):
                    st.image(product["image_url"], width=160)
            with details_column:
                st.markdown(f"**{product.get('title') or 'Product'}**")
                st.caption(f"ASIN: {product.get('parent_asin', 'Unknown')}")
                rating = product.get("rating")
                if rating is not None:
                    st.write(f"Rating: {rating:.1f}/5")


st.title("Amazon Products RAG Assistant")
st.caption("Answers are grounded in the available product catalog.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me about the available toys and games."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_products(message.get("products", []))

if prompt := st.chat_input("Ask about a product"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = requests.post(
                f"{config.API_URL.rstrip('/')}/rag/",
                json={"query": prompt},
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
                response_data = response.json()
                answer = response_data["answer"]
                products = response_data.get("products", [])
            except (KeyError, TypeError, ValueError) as exc:
                st.error(f"The API returned an invalid response: {exc}")
            else:
                st.markdown(answer)
                render_products(products)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "products": products}
                )
