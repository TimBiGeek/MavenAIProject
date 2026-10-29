from api.agents.retireval_generation import rag_pipline
from qdrant_client import QdrantClient
from langsmith import Client
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

from ragas.dataset_schema import SingleTurnSample

from ragas.metrics.collections import Faithfulness, AnswerRelevancy

from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall

ls_client = Client()
qdrant_client = QdrantClient(url="http://localhost:6333")



def build_ragas_clients(openai_client: AsyncOpenAI):

    ragas_llm = llm_factory("gpt-6-luna", client=openai_client, max_completion_tokens=4000)

    args = ragas_llm.model_args
    args.pop("max_tokens", None)
    args.pop("top_p", None)
    args["temperature"] = 0.0
    args["max_completion_tokens"] = 4000

    ragas_embeddings = embedding_factory("openai", model="text-embedding-3-small", client=openai_client)

    return ragas_llm, ragas_embeddings

async def ragas_faitfulness(run, example):

    async with AsyncOpenAI() as openai_client:

        ragas_llm, ragas_embeddings = build_ragas_clients(openai_client)

        scorer = Faithfulness(llm=ragas_llm)

        result = await scorer.ascore(
            user_input=run.outputs["question"],
            response=run.outputs["answer"],
            retrieved_contexts=run.outputs["retrieved_context"]
        )

    return result.value


async def ragas_relevancy(run, example):

    async with AsyncOpenAI() as openai_client:

        ragas_llm, ragas_embeddings = build_ragas_clients(openai_client)

        scorer = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)

        result = await scorer.ascore(
            user_input=run.outputs["question"],
            response=run.outputs["answer"]
        )

    return result.value

async def ragas_context_precision_id_based(run, example):

    sample = SingleTurnSample(
        retrieved_context_ids=run.outputs["retrieved_context_ids"],
        reference_context_ids=example.outputs["reference_context_ids"]
    )

    scorer = IDBasedContextPrecision()

    return await scorer.single_turn_ascore(sample)

async def ragas_context_recall_id_based(run, example):

    sample = SingleTurnSample(
        retrieved_context_ids=run.outputs["retrieved_context_ids"],
        reference_context_ids=example.outputs["reference_context_ids"]
    )

    scorer = IDBasedContextRecall()

    return await scorer.single_turn_ascore(sample)


result = ls_client.evaluate(
    lambda x: rag_pipline(x["question"], qdrant_client),
    data="rag-evaluation-dataset",
    evaluators=[
        ragas_faitfulness,
        ragas_relevancy,
        ragas_context_precision_id_based,
        ragas_context_recall_id_based
    ],
    experiment_prefix="rag",
    max_concurrency=10
)

