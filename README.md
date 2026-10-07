# LLM-Powered Research Assistant

A Retrieval-Augmented Generation (RAG) prototype that lets researchers upload scientific PDFs and ask questions in plain language. Answers are grounded in the uploaded papers and cite the paper title, authors and page.

Built as part of *AI Agents for Business Applications* (Prompt Engineering and RAG, Week 1). The case study uses four papers on prompt engineering (GPT-3, AutoPrompt, Prompt Tuning, Prompt Programming).

## How it works

1. **Load**: PDF text is extracted page by page with PyMuPDF.
2. **Chunk**: text is split into 512-token chunks (`tiktoken`, `cl100k_base`).
3. **Embed and index**: chunks are embedded and stored in an in-memory Chroma vector store.
4. **Retrieve**: the 5 chunks most similar to the question are fetched.
5. **Generate**: the chunks, each prefixed with a metadata header (title, authors, page), are sent to the LLM with a system prompt that restricts it to the provided context. If the context does not contain the answer, it replies "Sorry, this is out of my knowledge base."

| Component | Notebook (local) | Streamlit app (online) |
|---|---|---|
| Chat model | `openai/gpt-oss-20b` via LM Studio | `openai/gpt-oss-20b` via NVIDIA NIM |
| Embeddings | `nomic-embed-text-v1.5` via LM Studio | `nvidia/llama-3.2-nv-embedqa-1b-v1` via NVIDIA NIM |
| Vector store | Chroma | Chroma |
| Citation details | Curated dictionary for the four papers | PDF metadata, falling back to the file name |

## Repository contents

| File | Purpose |
|---|---|
| `LLM_Powered_Research_Assistant.ipynb` | Full walkthrough: baseline prompting, prompt engineering, RAG pipeline, and the deployment code |
| `app.py` | Streamlit app (upload PDFs, ask questions) |
| `requirements.txt` | Python dependencies |
| `.streamlit/secrets.toml.example` | Template for the API key secret |

## Run the Streamlit app locally

```bash
pip install -r requirements.txt
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then put your real key inside
streamlit run app.py
```

You need an NVIDIA NIM API key (starts with `nvapi-`) from [build.nvidia.com](https://build.nvidia.com).

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub. **Never commit `config.json` or `.streamlit/secrets.toml`**; both are in `.gitignore`.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and choose **Create app**.
3. Select the repo and branch, and set the main file to `app.py`.
4. In **Advanced settings**, choose Python 3.11 and add this under **Secrets**:
   ```toml
   API_KEY = "nvapi-your-real-key"
   ```
5. Click **Deploy**.

Optional secret: `API_BASE` overrides the default endpoint (`https://integrate.api.nvidia.com/v1`).

## Run the notebook locally

1. Start LM Studio's local server and load `openai/gpt-oss-20b` plus an embedding model (for example `nomic-embed-text-v1.5`). Set the context length to at least 8192.
2. Create a `config.json` with `OPENAI_API_KEY` and `OPENAI_API_BASE` (this file is git-ignored).
3. Put the PDFs in a `ResearchPaper/` folder and run the cells in order.

Note: `gpt-oss-20b` is a reasoning model, so reasoning tokens count toward `max_tokens`. The code uses generous limits (4000 for plain calls, 3000 for RAG) to avoid empty or truncated answers.

## Limitations

- Quality depends on the PDFs: scanned or badly formatted documents may extract poorly.
- Uploaded PDFs often lack reliable title and author metadata, so citations fall back to the file name.
- The vector store is in memory and is rebuilt whenever new files are uploaded.
- Answers are only as good as the retrieved chunks; deeper domain interpretation still needs human review.
