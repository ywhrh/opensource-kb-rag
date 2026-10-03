"""Read user-supplied questions and reference answers; never generate a test set."""
import json
from pathlib import Path
from typing import Literal
from pydantic import AliasChoices, BaseModel, Field
from core.config_loader import ROOT, digest


class GoldenCase(BaseModel):
    id: str
    category: Literal['semantic', 'keyword', 'distractor']
    question: str = Field(min_length=1, max_length=16000)
    reference_answer: str = Field(min_length=1, validation_alias=AliasChoices('reference_answer', 'reference', 'ground_truth'))


def load_dataset(path):
    path = Path(path)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_relative_to((ROOT / 'data').resolve()):
        raise ValueError('The golden dataset must be stored inside the project data directory')
    if not path.exists():
        raise ValueError('Golden dataset not found. Provide your questions and reference answers before evaluation.')
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError('Golden dataset must be a nonempty JSON array')
    cases = [GoldenCase.model_validate(row) for row in data]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError('Golden case IDs must be unique')
    if any(not case.question.strip() or not case.reference_answer.strip() for case in cases):
        raise ValueError('Every question and reference answer must be nonblank')
    return cases, digest([case.model_dump() for case in cases])
