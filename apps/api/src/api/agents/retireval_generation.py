import openai
import cohere
from pathlib import Path
from qdrant_client import QdrantClient
from langsmith import traceable, get_current_run_tree
import instructor
from pydantic import BaseModel, Field
from qdrant_client.models import Filter, FieldCondition, MatchValue, Prefetch, Document, models

from api.agents.utils.prompt_managment import prompt_template_config

class RAGUsedContext(BaseModel):
    id: str = Field(description="ID of the item used to answer the question")
    description: str = Field(description="Description of the item used to answer the question")

class RAGGenerationResponse(BaseModel):
    answer: str = Field(description="Answer to the question")
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question")

@traceable(
    name="embed_query",
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


@traceable(
    name="retrieve_data",
    run_type="retriever")
def retrieve_data(query, qdrant_client, collection_name="toys_and_games_1K", k=5,hybrid = True):
    query_embedding = get_embedding(query)
    if hybrid:
        search_result = qdrant_client.query_points(
                collection_name="toys_and_games_1K",
                prefetch=[
                    Prefetch(
                        query=query_embedding,
                        using="text-embedding-3-small",
                        limit=20
                    ),
                    Prefetch(
                        query=Document(
                            text=query,
                            model="qdrant/bm25"
                        ),
                        using="bm25",
                        limit=20
                    )
                ],
                query=models.RrfQuery(rrf=models.Rrf(weights=[3,1])),
                limit=k
            )
    else:
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


@traceable(
         name="rerank_data",
         run_type="tool")
def rerank_data(query, context, top_k=5):

    cohere_client = cohere.ClientV2()

    response = cohere_client.rerank(
        model="rerank-v5.0-pro",
        query=query,
        documents=context["retrieved_context"],
        top_n=top_k
    )

    order = [result.index for result in response.results]

    return {
        "retrieved_context_ids": [context["retrieved_context_ids"][i] for i in order],
        "retrieved_context": [context["retrieved_context"][i] for i in order],
        "similarity_scores": [context["similarity_scores"][i] for i in order],
        "retrieved_context_ratings": [context["retrieved_context_ratings"][i] for i in order]
    }


@traceable(
         name="format_retrieved_context",
         run_type="prompt")

def process_context(context):

    formatted_context = ""

    for id, chunk, rating in zip(context["retrieved_context_ids"], context["retrieved_context"], context["retrieved_context_ratings"]):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"

    return formatted_context

@traceable(
    name="build_prompt",
    run_type="prompt"
)
def build_prompt(preprocessed_context, query):
    prompt_path = Path(__file__).parent / "prompts" / "retrieval_generation.yaml"
    template = prompt_template_config(prompt_path, "retrieval_generation")
    return template.render(
        preprocessed_context=preprocessed_context,
        question=query,
    )

@traceable(
    run_type="llm",
    metadata={
        "ls_provider": "openai",
        "ls_model_name": "gpt-6-luna",
    },
)
def generate_answer(prompt):

    client = instructor.from_provider(
        "openai/gpt-6-luna",
        mode=instructor.Mode.RESPONSES_TOOLS
    )

    response, raw_response = client.create_with_completion(
        messages=[
            {"role": "system", "content": prompt}
        ],
        reasoning={"effort": "none"},
        response_model=RAGGenerationResponse
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.input_tokens,
            "output_tokens": raw_response.usage.output_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }

    return response

@traceable(name="rag_pipeline",
           run_type="chain"
           )
def rag_pipeline(query, qdrant_client, top_k=5, hybrid=True,rerank=True, retrieve_k=20):

     retrieved_context = retrieve_data(
        query,
        qdrant_client,
        k=retrieve_k if rerank else top_k,
        hybrid=hybrid
    )

     if rerank:
        retrieved_context = rerank_data(
            query,
            retrieved_context,
            top_k=top_k
        )

     all_context = process_context(retrieved_context)
     prompt = build_prompt(all_context, query)
     answer = generate_answer(prompt)

     final_answer = {
        "answer": answer.answer,
        "references": answer.references,
        "question": query,
        "retrieved_context_ids": retrieved_context["retrieved_context_ids"],
        "retrieved_context": retrieved_context["retrieved_context"]
    }

     return final_answer


def rag_pipeline_wraper(query, top_k=5):

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    result = rag_pipeline(query, qdrant_client, top_k)

    used_context = []

    for item in result.get("references", []):
        payload = qdrant_client.scroll(
            collection_name="toys_and_games_1K",
            with_payload=True,
            with_vectors=False,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="parent_asin",
                        match=MatchValue(value=item.id)
                    )
                ]
            )
        )[0][0].payload

        image_url = payload.get("images", "")
        price = payload.get("price")

        if image_url:
            used_context.append(
                {
                    "id": item.id,
                    "image_url": image_url,
                    "price": price,
                    "description": item.description
                }
            )

    return {
        "answer": result["answer"],
        "used_context": used_context
    }