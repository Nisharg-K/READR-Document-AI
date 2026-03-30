
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

import chromadb
import fitz
import ollama
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


MODEL_NAME = "phi3:medium"
EMBED_MODEL = "nomic-embed-text"
OCR_MODEL = "qwen3-vl:8b"
BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR.parent / "ui"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
MIN_EXTRACTED_TEXT_LENGTH = 50
SUMMARY_CONTEXT_CHARS = 12000
TOP_K = 5
OCR_RENDER_ZOOM = 1.5
MODEL_KEEP_ALIVE = "30m"
SUMMARY_OPTIONS = {"temperature": 0, "num_ctx": 4096, "num_predict": 300}
ANSWER_OPTIONS = {"temperature": 0, "num_ctx": 2048, "num_predict": 200}
OCR_OPTIONS = {"temperature": 0, "num_ctx": 1024, "num_predict": 1800}
EXHAUSTIVE_QUERY_TERMS = {"first", "last", "all", "every", "list", "which semester", "compare"}
EMBED_MODEL_KEYWORDS = ("embed", "embedding", "bert")

CHAT_HISTORY: dict[str, list[dict[str, str]]] = {}

chroma_client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},
)

app = FastAPI(title="READR")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    doc_id: str
    question: str
    model_name: str | None = None


class NewChatRequest(BaseModel):
    session_id: str = "default"


def model_response_items() -> list[Any]:
    response = ollama.list()
    if hasattr(response, "models"):
        return list(response.models)
    if isinstance(response, dict):
        return list(response.get("models", []))
    return []


def get_item_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def is_embedding_model(model_name: str, families: list[str]) -> bool:
    haystack = " ".join([model_name.lower(), *[family.lower() for family in families]])
    return any(keyword in haystack for keyword in EMBED_MODEL_KEYWORDS)


def list_local_chat_models() -> list[dict[str, str]]:
    models: list[dict[str, str]] = []
    seen: set[str] = set()

    for item in model_response_items():
        name = str(get_item_value(item, "model", "")).strip()
        if not name or name in seen:
            continue

        details = get_item_value(item, "details", {})
        family = str(get_item_value(details, "family", "")).strip()
        families = get_item_value(details, "families", []) or []
        families = [str(value).strip() for value in families if str(value).strip()]
        parameter_size = str(get_item_value(details, "parameter_size", "")).strip()

        if is_embedding_model(name, [family, *families]):
            continue

        label = name
        if parameter_size:
            label = f"{name} ({parameter_size})"

        seen.add(name)
        models.append(
            {
                "name": name,
                "label": label,
                "family": family or (families[0] if families else ""),
            }
        )

    return models


def resolve_chat_model(requested_model: str | None) -> str:
    available_models = list_local_chat_models()
    available_names = {item["name"] for item in available_models}
    if requested_model and requested_model in available_names:
        return requested_model
    if MODEL_NAME in available_names:
        return MODEL_NAME
    if available_models:
        return available_models[0]["name"]
    raise HTTPException(status_code=500, detail="No local Ollama chat models are available.")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as document:
            pages = [page.get_text("text") for page in document]
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {exc}") from exc
    return "\n".join(pages).strip()


def render_pdf_page_as_png(page: fitz.Page) -> bytes:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(OCR_RENDER_ZOOM, OCR_RENDER_ZOOM), alpha=False)
    return pixmap.tobytes("png")


def extract_text_from_scanned_pdf(file_bytes: bytes) -> str:
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as document:
            page_texts: list[str] = []
            for page_number, page in enumerate(document, start=1):
                image_bytes = render_pdf_page_as_png(page)
                response = ollama.chat(
                    model=OCR_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an OCR engine. Extract all readable text from the page image. "
                                "Return plain text only. Preserve line breaks when helpful. "
                                "Do not summarize, explain, or add commentary."
                            ),
                        },
                        {
                            "role": "user",
                            "content": "Extract the document text from this page image.",
                            "images": [image_bytes],
                        },
                    ],
                    options=OCR_OPTIONS,
                    keep_alive=MODEL_KEEP_ALIVE,
                )
                page_text = response.get("message", {}).get("content", "").strip()
                if page_text:
                    page_texts.append(f"--- Page {page_number} ---\n{page_text}")
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - depends on local Ollama availability
        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to OCR scanned PDF with {OCR_MODEL}. "
                "Make sure Ollama is running and the vision model is installed."
            ),
        ) from exc

    return "\n\n".join(page_texts).strip()


def extract_text(filename: str, file_bytes: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        try:
            return file_bytes.decode("utf-8").strip()
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1").strip()

    if suffix != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    text = extract_text_from_pdf(file_bytes)
    if len(text) >= MIN_EXTRACTED_TEXT_LENGTH:
        return text

    ocr_text = extract_text_from_scanned_pdf(file_bytes)
    if len(ocr_text) < MIN_EXTRACTED_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail="Could not extract enough text from the scanned PDF.",
        )
    return ocr_text


def normalize_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").splitlines()).strip()


def split_large_paragraph(paragraph: str, max_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    text = paragraph.strip()
    while start < len(text):
        end = min(start + max_size, len(text))
        if end < len(text):
            split_at = max(
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind(" ", start, end),
            )
            if split_at > start + 200:
                end = split_at + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    try:
        response = ollama.embed(model=EMBED_MODEL, input=texts)
    except Exception as exc:  # pragma: no cover - depends on local Ollama availability
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to generate embeddings with nomic-embed-text. "
                "Make sure Ollama is running and the model is installed."
            ),
        ) from exc
    embeddings = response.get("embeddings", [])
    if len(embeddings) != len(texts):
        raise HTTPException(status_code=500, detail="Embedding model returned invalid vectors.")
    return embeddings

def semantic_chunk(text: str, max_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    normalized = normalize_whitespace(text)
    paragraphs = [part.strip() for part in normalized.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(split_large_paragraph(paragraph, max_size, overlap))
            continue

        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_size:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())

        tail = current[-overlap:].strip() if current else ""
        current = f"{tail}\n\n{paragraph}".strip() if tail else paragraph
        if len(current) > max_size:
            chunks.extend(split_large_paragraph(current, max_size, overlap))
            current = ""

    if current:
        chunks.append(current.strip())

    return chunks





def embed_text(text: str) -> list[float]:
    embeddings = embed_texts([text])
    return embeddings[0] if embeddings else []


def call_ollama_json(document_text: str, model_name: str, filename: str = "") -> dict[str, Any]:
    prompt = f"""
Analyze the document below and return valid JSON with exactly this structure:
{{
  "summary": "paragraph",
  "entities": {{
    "names": [],
    "organisations": [],
    "dates": [],
    "values": []
  }},
  "insights": ["...", "...", "..."],
  "recommended_questions": ["...", "...", "..."]
}}


Rules for Summary:
- Stay grounded in the document only.
- Keep the summary to one concise paragraph IF AND ONLY IF THE CONTENT DESERVES ONE WHOLE PARAGRAPH, otherwise make it only one line or two, depending on the document, like a driving license or some single page docuemnt, just say what is the document type and what is the holder name.
- Return JSON only.

Rules for Entities:
- Stay grounded in the document only.
- Put only meaningful items in entities.
- For names, look for people, characters, or named individuals.
- dont add random single numbers found in document except dates and full serial numbers
- return JSON only.


Rules for Insights:
- Stay grounded in the document only.
- Insights should be short and specific.
- after normal insights, add those insights that most of people would miss and you find them maybe important to user.
- Return JSON only.

Rules for Recommended Questions:
- Stay grounded in the document only.
- Generate exactly 3 questions.
- Each question must be answerable directly from the document text provided here.
- Do not ask questions that require outside knowledge, assumptions, opinions, or missing context.
- Prefer questions about concrete facts in the document such as names, dates, amounts, identifiers, responsibilities, deadlines, sections, or key decisions.
- Make the questions useful for a user exploring this specific document.
- Keep each question short, clear, and specific.
- If the document is very small or limited, ask simple fact-based questions that can still be answered from the text.
- Return the questions in `recommended_questions` only.

Document filename:
{filename or "Unknown"}

Document:
{document_text[:SUMMARY_CONTEXT_CHARS]}
""".strip()

    try:
        response = ollama.chat(
            model=model_name,
            format="json",
            messages=[
                {
                    "role": "system",
                    "content": "You extract grounded summaries, entities, and insights from documents.",
                },
                {"role": "user", "content": prompt},
            ],
            options=SUMMARY_OPTIONS,
            keep_alive=MODEL_KEEP_ALIVE,
        )
    except Exception as exc:  # pragma: no cover - depends on local Ollama availability
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to contact Ollama at http://localhost:11434. "
                "Make sure Ollama is running and the required models are installed."
            ),
        ) from exc

    try:
        raw_content = response["message"]["content"]
    except KeyError as exc:
        raise HTTPException(status_code=500, detail="Model returned invalid JSON.") from exc

    parsed = parse_json_object(raw_content)
    return normalize_analysis(parsed)

def parse_json_object(raw_content: str) -> dict[str, Any] | None:
    import json
    import re

    if not raw_content:
        return None

    content = raw_content.strip()

    # ✅ 1. Direct JSON parsing
    try:
        return json.loads(content)
    except Exception:
        pass

    # ✅ 2. Extract JSON block from text (handles extra text from LLM)
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except Exception:
            pass

    # ✅ 3. Try fixing common issues (optional robustness)
    try:
        cleaned = content.replace("\n", " ").replace("\t", " ")
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass

    # ✅ FINAL → return None (so fallback logic runs)
    return None

def as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def normalize_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    entities = payload.get("entities")
    if not isinstance(entities, dict):
        entities = {}

    names = as_string_list(entities.get("names"))
    organisations = as_string_list(entities.get("organisations") or entities.get("orgs"))
    dates = as_string_list(entities.get("dates"))
    values = as_string_list(entities.get("values"))
    insights = as_string_list(payload.get("insights"))
    recommended_questions = as_string_list(
        payload.get("recommended_questions") or payload.get("recommended questions")
    )[:3]

    summary = payload.get("summary")
    summary_text = str(summary).strip() if isinstance(summary, str) else ""

    return {
        "summary": summary_text,
        "entities": {
            "names": names,
            "organisations": organisations,
            "dates": dates,
            "values": values,
        },
        "insights": insights,
        "recommended_questions": recommended_questions,
    }


def unique_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def collect_label_values(document_text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in document_text.splitlines():
        match = re.match(r"^\s*([A-Za-z][A-Za-z0-9 /()&,'-]{1,40})\s*:\s*(.+?)\s*$", line)
        if not match:
            continue
        label = re.sub(r"\s+", " ", match.group(1)).strip()
        value = re.sub(r"\s+", " ", match.group(2)).strip()
        if value:
            pairs.append((label, value))
    return pairs


def first_match(text: str, pattern: str) -> str:
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def build_fallback_analysis(document_text: str) -> dict[str, Any]:
    compact_text = re.sub(r"\s+", " ", document_text).strip()
    lines = [line.strip() for line in document_text.splitlines() if line.strip()]
    label_values = collect_label_values(document_text)

    summary_parts = []
    if lines:
        summary_parts.append(" ".join(lines[:3])[:320])
    if label_values:
        summary_parts.append(
            "Key fields include "
            + ", ".join(f"{label}: {value}" for label, value in label_values[:3])
            + "."
        )

    names = unique_preserving_order(
        [value for label, value in label_values if "name" in label.lower()]
        + re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", document_text)
        + re.findall(r"\b[A-Z][A-Z]+(?:\s+[A-Z][A-Z]+){1,4}\b", document_text)
    )[:8]
    organisations = unique_preserving_order(
        [
            value
            for label, value in label_values
            if any(
                token in label.lower()
                for token in ("company", "organization", "organisation", "institution", "department", "office")
            )
        ]
        + re.findall(
            r"\b(?:[A-Z][A-Za-z&()'-]+(?:\s+[A-Z][A-Za-z&()'-]+){1,6})\b",
            document_text,
        )
    )
    organisations = [
        item for item in organisations
        if any(
            keyword in item.lower()
            for keyword in (
                "inc",
                "llc",
                "ltd",
                "corp",
                "company",
                "organization",
                "organisation",
                "university",
                "college",
                "school",
                "institute",
                "department",
                "office",
                "agency",
            )
        )
    ][:8]

    dates = unique_preserving_order(
        re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", document_text)
        + re.findall(r"\b\d{4}-\d{2}-\d{2}\b", document_text)
        + re.findall(r"\b\d{4}-\d{2}\b", document_text)
        + re.findall(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[ ,.-]+\d{4}\b",
            document_text,
            re.IGNORECASE,
        )
    )[:8]
    values = unique_preserving_order(
        [f"{label}: {value}" for label, value in label_values[:8]]
        + re.findall(r"\b\d+(?:\.\d+)?%?\b", document_text)
    )[:8]

    insights = [f"{label}: {value}" for label, value in label_values[:5]]
    if lines:
        insights.extend(lines[:2])
    if not insights and compact_text:
        insights.append(compact_text[:160] + ("..." if len(compact_text) > 160 else ""))

    recommended_questions: list[str] = []
    if names:
        recommended_questions.append(f"Who is {names[0]} in this document?")
    if organisations:
        recommended_questions.append(f"What role does {organisations[0]} play in this document?")
    if dates:
        recommended_questions.append(f"What is important about the date {dates[0]} in this document?")
    if label_values:
        recommended_questions.append(f"What are the key details listed under {label_values[0][0]}?")
    if not recommended_questions:
        recommended_questions = [
            "What are the most important details in this document?",
            "Are there any deadlines, dates, or numbers I should pay attention to?",
            "What should I ask next to better understand this document?",
        ]

    return {
        "summary": " ".join(summary_parts).strip(),
        "entities": {
            "names": names,
            "organisations": organisations,
            "dates": dates,
            "values": values,
        },
        "insights": insights[:5],
        "recommended_questions": unique_preserving_order(recommended_questions)[:3],
    }


def merge_analysis_with_fallback(analysis: dict[str, Any], document_text: str) -> dict[str, Any]:
    fallback = build_fallback_analysis(document_text)
    merged_entities = {}
    for key in ("names", "organisations", "dates", "values"):
        primary = analysis.get("entities", {}).get(key, [])
        merged_entities[key] = primary if primary else fallback["entities"][key]

    return {
        "summary": analysis.get("summary") or fallback["summary"],
        "entities": merged_entities,
        "insights": analysis.get("insights") or fallback["insights"],
        "recommended_questions": analysis.get("recommended_questions") or fallback["recommended_questions"],
    }


def call_ollama_answer(question: str, context: str, model_name: str) -> str:
    system_prompt = (
        "You are READR, a document assistant.\n"
        "Answer ONLY from the context provided.\n"
        "Never use outside knowledge.\n"
        "Quote exact values when available.\n"
        "If not found say: Not found in document.\n"
        "Be concise. Maximum 4 sentences.\n"
        "For lists use bullet points."
    )
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{question}",
                },
            ],
            options=ANSWER_OPTIONS,
            keep_alive=MODEL_KEEP_ALIVE,
        )
    except Exception as exc:  # pragma: no cover - depends on local Ollama availability
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to contact Ollama at http://localhost:11434. "
                "Make sure Ollama is running and the required models are installed."
            ),
        ) from exc

    answer = response.get("message", {}).get("content", "").strip()
    return answer or "Not found in document."


def clear_existing_doc(doc_id: str) -> None:
    try:
        existing = collection.get(where={"doc_id": doc_id})
    except Exception:
        return
    ids = existing.get("ids", [])
    if ids:
        collection.delete(ids=ids)


def store_document(doc_id: str, chunks: list[str]) -> None:
    embeddings = embed_texts(chunks)
    metadatas = []
    ids = []
    for index, chunk in enumerate(chunks):
        metadatas.append(
            {
                "doc_id": doc_id,
                "chunk_index": index,
            }
        )
        ids.append(f"{doc_id}_chunk_{index}")

    clear_existing_doc(doc_id)
    try:
        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store document in ChromaDB: {exc}") from exc


def should_fetch_all_chunks(question: str) -> bool:
    normalized = question.lower()
    return any(term in normalized for term in EXHAUSTIVE_QUERY_TERMS)


def sort_documents_by_chunk_index(results: dict[str, Any]) -> list[str]:
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    paired = sorted(
        zip(documents, metadatas),
        key=lambda item: item[1].get("chunk_index", 0),
    )
    return [document for document, _metadata in paired]


def retrieve_context(doc_id: str, question: str) -> str:
    if should_fetch_all_chunks(question):
        results = collection.get(where={"doc_id": doc_id})
        documents = sort_documents_by_chunk_index(results)
    else:
        question_embedding = embed_text(question)
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=TOP_K,
            where={"doc_id": doc_id},
        )
        documents = list(results.get("documents", [[]])[0])

    if not documents:
        return ""
    return "\n\n".join(documents)


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    model_name: str | None = Form(default=None),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name.")

    selected_model = resolve_chat_model(model_name)
    file_bytes = await file.read()
    document_text = extract_text(file.filename, file_bytes)
    if not document_text:
        raise HTTPException(status_code=400, detail="The uploaded document is empty.")

    chunks = semantic_chunk(document_text)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Could not create meaningful chunks from the uploaded document.",
        )

    doc_id = str(uuid4())
    store_document(doc_id, chunks)
    analysis = merge_analysis_with_fallback(
        call_ollama_json(document_text, selected_model, file.filename),
        document_text,
    )

    entities = analysis.get("entities", {})
    return {
        "doc_id": doc_id,
        "summary": analysis.get("summary", ""),
        "entities": {
            "names": entities.get("names", []),
            "organisations": entities.get("organisations", []),
            "dates": entities.get("dates", []),
            "values": entities.get("values", []),
        },
        "insights": analysis.get("insights", []),
        "recommended_questions": analysis.get("recommended_questions", []),
        "word_count": len(document_text.split()),
        "chunk_count": len(chunks),
        "model_name": selected_model,
    }


@app.post("/query")
async def query_document(payload: QueryRequest) -> dict[str, str]:
    question = payload.question.strip()
    doc_id = payload.doc_id.strip()
    selected_model = resolve_chat_model(payload.model_name)
    if not doc_id:
        raise HTTPException(status_code=400, detail="doc_id is required.")
    if not question:
        raise HTTPException(status_code=400, detail="question is required.")

    existing = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
    if not existing.get("ids"):
        raise HTTPException(status_code=404, detail="Document not found.")

    context = retrieve_context(doc_id, question)
    answer = "Not found in document." if not context else call_ollama_answer(question, context, selected_model)
    return {"answer": answer}


@app.post("/new-chat")
async def new_chat(payload: NewChatRequest | None = None) -> dict[str, str]:
    session_id = payload.session_id if payload else "default"
    CHAT_HISTORY[session_id] = []
    return {"status": "cleared"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
async def get_models() -> dict[str, Any]:
    models = list_local_chat_models()
    return {
        "models": models,
        "default_model": resolve_chat_model(None),
        "ocr_model": OCR_MODEL,
    }


app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
