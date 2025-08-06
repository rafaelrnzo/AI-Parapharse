from pydantic import BaseModel
from typing import List

class CorrectionRequest(BaseModel):
    text: str

class CorrectionResponse(BaseModel):
    corrected: str
    tokenized: List[str]
    short_words: List[str]
    pronomina_mixed: bool
    typo_words: List[str]  # <- tambahkan ini
