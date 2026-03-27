
# READR
AI Document Processing & Insight Extractor

---

## What it does

Upload a PDF or TXT file. READR reads it, extracts a summary,
key entities, and insights. Then lets you ask questions about
the document in a chat interface.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JS |
| Backend | FastAPI (Python) |
| Vector Store | ChromaDB |
| LLM Server | Ollama |
| Chat + Summary + Insights | LLaMA 3.1 8B Instruct Q4_K_M |
| Embeddings | Nomic Embed Text |
| OCR (under development) | Qwen3 VL 8B |

---

## Prerequisites

- Python 3.10+
- Ollama installed — https://ollama.com

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/READR.git
cd READR
```

**2. Pull required models**
```bash
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull nomic-embed-text
```

**3. Install dependencies**
```bash
cd backend
python -m pip install -r requirements.txt
```

**4. Run**
```bash
python server.py
```

**5. Open browser**
```
http://localhost:8000
```

To use READR from other devices on the same network, start the server on all interfaces:
```bash
set READR_HOST=0.0.0.0
python server.py
```

Then open one of the URLs returned by:
```bash
curl http://localhost:8000/server-info
```

Example:
```
http://192.168.1.25:8000
```

Optional environment variables:
```bash
set READR_PORT=8000
set READR_SERVER_NAME=READR-HOST
set READR_ALLOWED_ORIGINS=http://192.168.1.50:8000,http://192.168.1.51:8000
```

---

## Requirements

```
fastapi
uvicorn
ollama
pymupdf
python-multipart
chromadb
rank_bm25
```

---

## Project Structure

```
READR/
├── backend/
│   ├── server.py          
│   ├── requirements.txt   
│   └── chroma_db/         
└── ui/
    ├── index.html         
    ├── style.css          
    └── app.js             
```

---

## How it works

**On upload:**
```
PDF or TXT
    ↓
Extract text (PyMuPDF)
    ↓
Split into chunks
    ↓
Embed chunks (nomic-embed-text)
    ↓
Store in ChromaDB(Vector DB)
    ↓
Generate summary, entities, insights (LLaMA)
```

**On every chat message:**
```
User question
    ↓
Embed question (nomic-embed-text)
    ↓
Find closest chunks in ChromaDB
    ↓
Send chunks + question to LLaMA
    ↓
Answer grounded in document
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | /upload | Upload PDF or TXT, returns summary, entities, insights |
| POST | /query | Ask a question, returns answer from document |
| POST | /new-chat | Clear chat history |
| POST | /connect-device | Register a device and return reachable server URLs |
| GET | /health | Server status |
| GET | /server-info | Return LAN URLs and currently connected devices |

