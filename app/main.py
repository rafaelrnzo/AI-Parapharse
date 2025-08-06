from fastapi import FastAPI
from models import CorrectionRequest, CorrectionResponse
from rag_engine import grammar_corrector

app = FastAPI(title="AI Grammarly Bahasa Indonesia")

@app.post("/correct", response_model=CorrectionResponse)
async def correct_grammar(request: CorrectionRequest):
    result = grammar_corrector(request.text)
    return CorrectionResponse(**result)
