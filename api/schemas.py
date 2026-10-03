"""API contracts. Provider credentials are never included in experiment snapshots."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from core.config_loader import RAGConfig


class Credentials(BaseModel):
    model_config = ConfigDict(extra='forbid')
    openai_api_key: SecretStr | None = None
    cohere_api_key: SecretStr | None = None


class ConfigUpdate(BaseModel):
    config: RAGConfig
    expected_revision: str | None = None


class RunRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    query: str = Field(min_length=1, max_length=16000)
    source_id: str = Field(default='project', pattern=r'^(project|upload_[a-f0-9]{32})$')
    config_snapshot: RAGConfig | None = None
    credentials: Credentials = Field(default_factory=Credentials)
    persist_history: bool = True
    index_collection: str | None = Field(default=None, pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    config_snapshot: RAGConfig | None = None
    source_id: str = Field(default='project', pattern=r'^(project|upload_[a-f0-9]{32})$')
    credentials: Credentials = Field(default_factory=Credentials)
    modes: list[Literal['vector_only', 'hybrid', 'hybrid_rerank']] = Field(default_factory=lambda: ['vector_only', 'hybrid', 'hybrid_rerank'], min_length=1)


class IngestRequest(BaseModel):
    source_id: str = Field(default='project', pattern=r'^(project|upload_[a-f0-9]{32})$')
    config_snapshot: RAGConfig | None = None
    credentials: Credentials = Field(default_factory=Credentials)


class LoadTestRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    config_snapshot: RAGConfig | None = None
    source_id: str = Field(default='project', pattern=r'^(project|upload_[a-f0-9]{32})$')
    queries: list[str] = Field(min_length=1, max_length=1000)
    credentials: Credentials = Field(default_factory=Credentials)
