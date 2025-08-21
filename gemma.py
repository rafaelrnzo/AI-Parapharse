import os
import json
import asyncio
from typing import List, Literal
import nltk
import httpx
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from difflib import SequenceMatcher
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
from better_profanity import profanity

load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL") or "http://192.168.100.3:11500"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL") or "gemma3:latest"
REDIS_URL = os.getenv("REDIS_URL") or "redis://localhost:6379/0"

OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"

try:
    _ = nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

app = FastAPI(title="Ollama Grammar Corrector + Redis Cache")

redis_client = redis.from_url(REDIS_URL, decode_responses=True)

profanity.load_censor_words()
additional_bad = [
    "tai", "goblok", "kontol", "anjing", "bangsat", "sialan", "memek",
    "kampret", "tolol", "brengsek", "bajingan", "setan", "keparat"
]
existing_normalized = [str(w).lower() for w in profanity.CENSOR_WORDSET]
merged_set = set(existing_normalized) | set(w.lower() for w in additional_bad)
profanity.CENSOR_WORDSET = list(merged_set)

StyleType = Literal["formal", "casual", "santai"]

class CorrectionRequest(BaseModel):
    text: str
    style: StyleType = "formal"

class CorrectionResponse(BaseModel):
    corrected: str
    tokenized: List[str]
    short_words: List[str]
    pronomina_mixed: bool
    typo_words: List[str]

class ProfanityResponse(BaseModel):
    censored: str
    found: bool
    bad_words: List[str]

def build_prompt(text: str, style: StyleType) -> str:
    if style == "formal":
        style_desc = "Gaya penulisan profesional dan sesuai kaidah bahasa Indonesia yang baku"
    elif style == "casual":
        style_desc = "Gaya penulisan santai namun tetap sopan, seperti obrolan sehari-hari"
    else:
        style_desc = "Gaya penulisan bebas, seperti percakapan dengan teman dekat"

    return (
        f"Perbaiki dan parafrase kalimat berikut agar sesuai dengan {style_desc}.\n"
        f"Kalimat asli:\n{text}\n"
        "Jawaban (hanya kalimat hasil koreksi, tanpa penjelasan):"
    )

def detect_typos(orig: List[str], corrected: List[str]) -> List[str]:
    matcher = SequenceMatcher(None, orig, corrected)
    typos = []
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            typos.extend(orig[i1:i2])
    return list({t for t in typos if t.isalpha()})

async def call_ollama(prompt: str) -> str:
    headers = {"Content-Type": "application/json"}
    body = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 1000
        }
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(OLLAMA_URL, headers=headers, json=body)
    if resp.status_code != 200:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise HTTPException(status_code=resp.status_code, detail={"ollama_error": detail})
    data = resp.json()
    return data.get("response", "").strip()

@app.post("/correct", response_model=CorrectionResponse)
async def correct_grammar(request: CorrectionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="`text` must be non-empty.")

    cache_key = f"grammar:{request.style}:{request.text.strip().lower()}"
    cached = await redis_client.get(cache_key)
    if cached:
        return CorrectionResponse(**json.loads(cached))

    prompt = build_prompt(request.text, request.style)
    corrected = await call_ollama(prompt)

    try:
        orig_tokens = word_tokenize(request.text)
        corrected_tokens = word_tokenize(corrected)
    except Exception:
        orig_tokens = request.text.split()
        corrected_tokens = corrected.split()

    short_words = [w for w in corrected_tokens if len(w) == 1 and w.isalpha()]
    lowered_corrected = [w.lower() for w in corrected_tokens]
    pronomina_mixed = ("aku" in lowered_corrected and "kami" in lowered_corrected)
    typo_words = detect_typos(orig_tokens, corrected_tokens)

    result = CorrectionResponse(
        corrected=corrected,
        tokenized=corrected_tokens,
        short_words=short_words,
        pronomina_mixed=pronomina_mixed,
        typo_words=typo_words
    )

    await redis_client.set(cache_key, json.dumps(result.dict()), ex=3600)

    return result

@app.post("/profanity", response_model=ProfanityResponse)
async def detect_profanity(request: CorrectionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="`text` must be provided.")
    
    cache_key = f"profanity:{request.text.strip().lower()}"
    cached = await redis_client.get(cache_key)
    if cached:
        return ProfanityResponse(**json.loads(cached))

    lowered = request.text.lower()
    try:
        tokens = word_tokenize(lowered)
    except Exception:
        tokens = lowered.split()
    bad_words_set = {word for word in tokens if profanity.contains_profanity(word)}
    found = len(bad_words_set) > 0
    censored = profanity.censor(request.text) if found else request.text

    result = ProfanityResponse(
        censored=censored,
        found=found,
        bad_words=sorted(list(bad_words_set))
    )

    await redis_client.set(cache_key, json.dumps(result.dict()), ex=3600)
    return result

@app.get("/healthz")
def healthz():
    return {"status": "ok", "ollama_url": OLLAMA_BASE_URL, "model": OLLAMA_MODEL}

@app.on_event("startup")
async def startup_event():
    await redis_client.ping()
    print("✅ Connected to Redis")

@app.on_event("shutdown")
async def shutdown_event():
    await redis_client.close()
