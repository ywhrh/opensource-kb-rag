"""Run genuine Ragas metrics with an explicit local or key-gated judge."""
import asyncio
import math
import os

from ragas import evaluate, EvaluationDataset
from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference, Faithfulness, AnswerCorrectness
from ragas.run_config import RunConfig

from core.models import embed, provider_key


class StudioEmbeddings(BaseRagasEmbeddings):
    def __init__(self, config, credentials):
        super().__init__()
        self.config = config
        self.credentials = credentials

    def embed_documents(self, texts):
        return embed(texts, self.config, self.credentials).tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts):
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text):
        return await asyncio.to_thread(self.embed_query, text)


def score_cases(rows, config, credentials):
    if not config.evaluation.metrics:
        raise ValueError('Select at least one Ragas metric')
    run_config = RunConfig(timeout=600, max_retries=2, max_workers=1, seed=config.generation.seed)
    if config.evaluation.judge_model == 'gpt-4o-mini':
        from langchain_openai import ChatOpenAI
        judge = ChatOpenAI(model='gpt-4o-mini', api_key=provider_key(credentials, 'openai'),
                           temperature=0, max_tokens=config.evaluation.judge_max_tokens)
    else:
        from langchain_ollama import ChatOllama
        judge = ChatOllama(model='llama3.1:8b', base_url=os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434'),
                           temperature=0, seed=config.generation.seed, num_ctx=16384,
                           num_predict=config.evaluation.judge_max_tokens)
    llm = LangchainLLMWrapper(judge, run_config=run_config)
    embeddings = StudioEmbeddings(config, credentials)
    available = {'context_recall': LLMContextRecall(),
                 'context_precision': LLMContextPrecisionWithReference(name='context_precision'),
                 'faithfulness': Faithfulness(), 'answer_correctness': AnswerCorrectness()}
    result = evaluate(EvaluationDataset.from_list(rows), metrics=[available[key] for key in config.evaluation.metrics],
                      llm=llm, embeddings=embeddings, run_config=run_config, raise_exceptions=False,
                      show_progress=False, allow_nest_asyncio=False)
    cleaned = [{key: float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None
                for key, value in scores.items()} for scores in result.scores]
    metrics, counts = {}, {}
    for key in config.evaluation.metrics:
        valid = [scores[key] for scores in cleaned if scores.get(key) is not None]
        metrics[key] = sum(valid) / len(valid) if valid else None
        counts[key] = len(valid)
    return metrics, cleaned, counts
