import openai
from qdrant_client import QdrantClient
from langsmith import traceable, get_current_run_tree

@traceable(
    run_type="embedding",
    metadata={
        "ls_provider": "openai",
        "ls_model_name": "text-embedding-3-small",
    },
)
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=text,
        model=model,
    )

    run_tree = get_current_run_tree()
    if run_tree is not None and response.usage is not None:
        run_tree.set(
            usage_metadata={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": 0,
                "total_tokens": response.usage.total_tokens,
            }
        )

    return response.data[0].embedding


@traceable(run_type="retriever")
def retrieve_data(query, qdrant_client, collection_name="toys_and_games_small_sample", k=5):
    query_embedding = get_embedding(query)
    search_result = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=k
    )
    retrieved_context_id = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_ratings = []
    products = []

    for result in search_result.points:
        payload = result.payload
        product_id = payload["parent_asin"]
        description = payload["all_description"]
        rating = payload["average_rating"]
        images = payload.get("images")

        if isinstance(images, list):
            images = images[0] if images else None
        if isinstance(images, dict):
            image_url = images.get("large") or images.get("hi_res") or images.get("thumb")
        else:
            image_url = images

        retrieved_context_id.append(product_id)
        retrieved_context.append(description)
        similarity_scores.append(result.score)
        retrieved_context_ratings.append(rating)
        products.append(
            {
                "parent_asin": product_id,
                "title": payload.get("title") or description[:120].strip(),
                "image_url": image_url,
                "rating": rating,
                "similarity": result.score,
            }
        )

    return {
        "retrieved_context_ids": retrieved_context_id,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
        "retrieved_context_ratings": retrieved_context_ratings,
        "products": products,
    }

@traceable(run_type="chain")
def process_context(retrieved_context):
    formatted_context = []

    for index, (product_id, description, rating, score) in enumerate(
        zip(
            retrieved_context["retrieved_context_ids"],
            retrieved_context["retrieved_context"],
            retrieved_context["retrieved_context_ratings"],
            retrieved_context["similarity_scores"],
            strict=True,
        ),
        start=1,
    ):
        formatted_context.append(
            f"Context {index}\n"
            f"Product ID: {product_id}\n"
            f"Rating: {rating}/5\n"
            f"Similarity: {score:.3f}\n"
            f"Description: {description}"
        )

    return "\n\n---\n\n".join(formatted_context)

@traceable(run_type="prompt")
def build_prompt(preprocessed_context, query):
    prompt = f"""
You are a shopping assistant that can answer question about the products in stock.
You will be given a question a nd a list of context.
Instruction:
 - Answer the question based on the context provided onlym
 - Never use word context and refer to it as the available products
 - Do not use markdown formatting.

 Context:
{preprocessed_context}
Question:
{query}

 """
    return prompt

@traceable(
    run_type="llm",
    metadata={
        "ls_provider": "openai",
        "ls_model_name": "gpt-6-luna",
    },
)
def generate_asnwer(prompt):
    response = openai.chat.completions.create(
        model="gpt-6-luna",
        messages=[
            {"role": "system", "content": prompt}
        ],
    )

    run_tree = get_current_run_tree()
    if run_tree is not None and response.usage is not None:
        run_tree.set(
            usage_metadata={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        )

    return response.choices[0].message.content

@traceable(run_type="chain")
def rag_pipline(query,qdrant_client, k=5):
    retrieved_context = retrieve_data(query,qdrant_client,collection_name="toys_and_games_small_sample", k=k)
    preprocessed_context=process_context(retrieved_context)
    prompt = build_prompt(preprocessed_context, query)
    answer = generate_asnwer(prompt)

    final_answer = {
      "answer":answer,
      "question":query,
      "retrieved_context_ids": retrieved_context["retrieved_context_ids"],
      "retrieved_context": retrieved_context["retrieved_context"],
      "products": retrieved_context["products"]
    }
    return final_answer