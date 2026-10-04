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


class MetricScorer:
    """Reuse judge/embeddings while returning one case's one metric at a time."""

    def __init__(self, config, credentials):
        self.run_config = RunConfig(timeout=600, max_retries=2, max_workers=1, seed=config.generation.seed)
        if config.evaluation.judge_model == 'gpt-4o-mini':
            from langchain_openai import ChatOpenAI
            judge = ChatOpenAI(model='gpt-4o-mini', api_key=provider_key(credentials, 'openai'),
                              temperature=0, max_tokens=config.evaluation.judge_max_tokens)
        else:
            from langchain_ollama import ChatOllama
            # Ragas prompts expect JSON; ordinary conversational output is not a score.
            judge = ChatOllama(model='llama3.1:8b', base_url=os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434'),
                               temperature=0, seed=config.generation.seed, num_ctx=16384, format='json',
                               num_predict=config.evaluation.judge_max_tokens)
        self.llm = LangchainLLMWrapper(judge, run_config=self.run_config)
        self.embeddings = StudioEmbeddings(config, credentials)

    def score(self, row, metric_name):
        available = {'context_recall': LLMContextRecall,
                     'context_precision': lambda: LLMContextPrecisionWithReference(name='context_precision'),
                     'faithfulness': Faithfulness, 'answer_correctness': AnswerCorrectness}
        result = evaluate(EvaluationDataset.from_list([row]), metrics=[available[metric_name]()],
                          llm=self.llm, embeddings=self.embeddings, run_config=self.run_config,
                          raise_exceptions=True, show_progress=False, allow_nest_asyncio=False)
        value = float(result.scores[0][metric_name])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Judge did not return a valid score between 0 and 1')
        return value
