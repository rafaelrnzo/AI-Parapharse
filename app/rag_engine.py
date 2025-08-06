from nltk.tokenize import word_tokenize
from difflib import SequenceMatcher
from vectorstore import retriever
from utils.prompt import build_prompt
from langchain.llms import Ollama

llm = Ollama(
    model="gemma3:latest",
    base_url="http://192.168.100.3:11434",
    temperature=0.3
)

def clean_result(text: str) -> str:
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    elif text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    return text.strip()

def detect_typos(original_tokens, corrected_tokens):
    typos = []
    matcher = SequenceMatcher(None, original_tokens, corrected_tokens)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            typos.extend(original_tokens[i1:i2])
    return [w for w in typos if w.isalpha()]

def grammar_corrector(text: str) -> dict:
    docs = retriever.get_relevant_documents(text)
    context = "\n".join([d.page_content for d in docs])
    prompt = build_prompt(text, context)
    raw_result = llm(prompt).strip()
    result = clean_result(raw_result)

    original_tokens = word_tokenize(text)
    corrected_tokens = word_tokenize(result)

    return {
        "corrected": result,
        "tokenized": corrected_tokens,
        "short_words": [w for w in corrected_tokens if len(w) == 1 and w.isalpha()],
        "pronomina_mixed": "Aku" in corrected_tokens and "kami" in corrected_tokens,
        "typo_words": detect_typos(original_tokens, corrected_tokens)
    }
