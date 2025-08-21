# grammar_checker_app.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from langchain_community.llms import Ollama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional

class GrammarCorrection(BaseModel):
    corrected: str = Field(description="Kalimat hasil perbaikan, TANPA penjelasan atau kutipan")

llm = Ollama(
    model="adijayainc/bhsa-llama3.2:latest",
    base_url="http://192.168.100.3:11500",
    temperature=0.3,
)

parser = JsonOutputParser(pydantic_object=GrammarCorrection)

prompt = PromptTemplate.from_template(
    """{context}

Kalimat asli:
{text}

Perbaiki kalimat di atas dengan ketentuan:
- Perbaiki EJAAN yang salah.
- Gunakan struktur tata bahasa Indonesia yang benar.
- Gunakan kata ganti orang yang konsisten.
- Jangan menambahkan kata baru.
- Jangan ganti makna kalimat.
- Tulis HANYA hasil kalimat final — TANPA penjelasan, TANPA tanda kutip.

Kalimat final dalam format JSON (misal: {{ "corrected": "..." }}):
{format_instructions}
"""
)

grammar_chain = (
    {"text": RunnablePassthrough(), "context": lambda _: "Berikut adalah referensi tata bahasa Indonesia yang benar:"}
    | RunnableLambda(lambda x: prompt.format_prompt(**x, format_instructions=parser.get_format_instructions()).to_string())
    | llm
    | parser
)

app = FastAPI(title="Indo Grammar Checker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("./", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "output": None})

@app.post("/correct", response_class=HTMLResponse)
async def correct(request: Request):
    form = await request.form()
    text = form.get("text")
    result = grammar_chain.invoke(text)
    return templates.TemplateResponse("index.html", {"request": request, "output": result.corrected, "input": text})
