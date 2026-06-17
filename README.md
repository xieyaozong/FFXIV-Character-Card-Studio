# FFXIV Multimodal RAG Assistant

Personal Final Fantasy XIV note assistant with retrieval citations and optional screenshot context.

## Flow

```text
User Question
  -> Retriever searches personal FF14 notes
  -> Relevant chunks + citations
  -> LLM-style answer
```

Optional screenshot path:

```text
Screenshot
  -> VLM caption / UI extraction
  -> RAG context
  -> Answer with visual context
```

## Data Policy

Do not copy large sections of official wikis, guides, or third-party攻略 sites into this repository. Keep personal notes in `docs_source/`, or add download/link scripts that point users to official sources.

## MVP

- Markdown notes to local vector index.
- Question answering with citations.
- Refuse to answer when no source is relevant.
- Docker-ready FastAPI and Streamlit entry points.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python scripts/build_index.py
streamlit run app/ui/streamlit_app.py
```

CLI:

```powershell
python scripts/run_chat.py "How should I prepare for raid mitigation?"
```

API:

```powershell
uvicorn app.main:app --reload
```

## Docker

```powershell
docker compose up --build
```

## Layout

```text
ffxiv-multimodal-rag-assistant/
  app/          FastAPI routes and Streamlit UI
  rag/          document loading, chunking, retrieval, answer/citation formatting
  vlm/          screenshot preprocessing and multimodal prompt hooks
  docs_source/  personal sample notes
  vector_db/    local generated index
  sample_data/  demo queries and synthetic screenshot folder
  scripts/      index, chat, evaluation, cleanup
  eval/         retrieval evaluation fixtures
  infra/        AWS and Kubernetes notes/manifests
  docs/         architecture and limitation notes
```

## Notes

The current index uses a local TF-IDF retriever so it runs without API keys. Swap `rag/embedder.py` and `rag/vector_store.py` for OpenAI embeddings, Chroma, FAISS, LangChain, or LlamaIndex when needed.
