import streamlit as st
import os
import tempfile
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from openai import OpenAI


# NVIDIA NIM exposes an OpenAI-compatible API
def get_setting(name, default=None):
    # Streamlit Cloud secrets (also exposed as env vars) first, then plain environment variables
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:                                                           # No secrets file configured (e.g. running locally)
        pass
    return os.environ.get(name, default)


NIM_API_KEY = get_setting("API_KEY")                                            # NVIDIA NIM key (nvapi-...), kept in Streamlit secrets
NIM_API_BASE = get_setting("API_BASE", "https://integrate.api.nvidia.com/v1")
LLM_MODEL = "openai/gpt-oss-20b"                                                # Chat model served by NVIDIA NIM
# Embedding models to try in order: the first one that your NIM account can actually call is used
EMBEDDING_CANDIDATES = [
    "nvidia/llama-3.2-nv-embedqa-1b-v1",
    "nvidia/nv-embedqa-e5-v5",
    "nvidia/nv-embedqa-mistral-7b-v2",
    "nvidia/nv-embed-v1",
    "baai/bge-m3",
    "snowflake/arctic-embed-l",
    "nvidia/embed-qa-4",
]

st.title("LLM-Powered Research Assistant")

# Stop with a clear message if the key is missing (OpenAI() would otherwise crash with a redacted error)
if not NIM_API_KEY:
    st.error("API_KEY is not set. In Streamlit Cloud open Manage app > Settings > Secrets and add: API_KEY = \"nvapi-...\"")
    st.stop()

# Initialize OpenAI-compatible client pointing at NVIDIA NIM
client = OpenAI(api_key=NIM_API_KEY, base_url=NIM_API_BASE)


class NIMEmbeddings(Embeddings):
    """Embeddings via NVIDIA NIM. The e5 retrieval models need input_type='passage' for documents and 'query' for questions."""

    def __init__(self, model: str, batch_size: int = 32):
        self.model = model
        self.batch_size = batch_size

    def _embed(self, texts: List[str], input_type: str) -> List[List[float]]:
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            result = client.embeddings.create(
                model=self.model,
                input=batch,
                encoding_format="float",
                extra_body={"input_type": input_type, "truncate": "END"},
            )
            vectors.extend(item.embedding for item in result.data)
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts, "passage")

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], "query")[0]


# Define the system prompt for the model
qna_system_message = """
You are an AI assistant designed to support research teams in efficiently reviewing scientific literature. Your task is to provide evidence-based, concise, and relevant summaries based on the context provided from research papers.

User input will include the necessary context for you to answer their questions. This context will begin with the token:

###Context
The context contains excerpts from one or more research papers, along with associated metadata such as titles, authors, abstracts, keywords, and specific sections relevant to the query.

When crafting your response
-Use only the provided context to answer the question.
-If the answer is found in the context, respond with concise and insight-focused summaries.
-Cite every source using the metadata header that precedes each excerpt: the paper title, authors, and page number (including the venue/journal and year when they are in the header). Never invent details; write "not provided" for anything missing.
-If the question is unrelated to the context or the context is empty, clearly respond with: "Sorry, this is out of my knowledge base."


Please adhere to the following response guidelines:
-Provide clear, direct answers using only the given context.
-Do not include any additional information outside of the context.
-Avoid rephrasing or generalizing unless explicitly relevant to the question.
-If no relevant answer exists in the context, respond with: "Sorry, this is out of my knowledge base."
-If the context is not provided, your response should also be: "Sorry, this is out of my knowledge base."


Here is an example of how to structure your response:

Answer:
[Answer based on context]

Source:
[Source details with journal name, title, authors, page number or section]
"""

# Define the user message template
qna_user_message_template = """
###Context
Here are some excerpts from research papers, each preceded by a metadata header (title, authors, source file, page), that are relevant to the question mentioned below:
{context}

###Question
{question}
"""


@st.cache_resource
def pick_embedding_model():
    # Listing a model on build.nvidia.com does not guarantee that this account can call it, so test each one
    errors = []
    for name in EMBEDDING_CANDIDATES:
        try:
            client.embeddings.create(model=name, input=["test"], encoding_format="float",
                                     extra_body={"input_type": "query", "truncate": "END"})
            return name
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__} {str(e)[:150]}")
    raise RuntimeError("No embedding model is available for this API key:\n" + "\n".join(errors))


@st.cache_resource
def load_and_process_pdfs(uploaded_files):
    all_documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        loader = PyMuPDFLoader(tmp_file_path)
        documents = loader.load()
        os.remove(tmp_file_path)                                                # Clean up the temporary file
        for doc in documents:
            # Keep citation details on every page (and later every chunk); fall back to the file name
            doc.metadata["source"] = uploaded_file.name
            doc.metadata["title"] = doc.metadata.get("title") or os.path.splitext(uploaded_file.name)[0]
            doc.metadata["author"] = doc.metadata.get("author") or "Unknown"
        all_documents.extend(documents)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=512,
    )
    document_chunks = text_splitter.split_documents(all_documents)

    # Create an in-memory vector store (or use a persistent one if needed)
    vectorstore = Chroma.from_documents(
        document_chunks,
        NIMEmbeddings(pick_embedding_model())
    )
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})


def generate_rag_response(user_input, retriever, max_tokens=3000, temperature=0.3, top_p=0.95):
    # Retrieve relevant document chunks
    relevant_document_chunks = retriever.invoke(user_input)

    # Prefix each chunk with its metadata so the model can cite title, authors and page
    context_list = []
    for d in relevant_document_chunks:
        m = d.metadata
        header = (f"[Title: {m.get('title') or m.get('source')} | Authors: {m.get('author') or 'Unknown'} | "
                  f"Source file: {m.get('source')} | Page: {int(m.get('page', 0)) + 1}]")
        context_list.append(header + "\n" + d.page_content)

    # Combine document chunks into a single context
    context_for_query = "\n\n".join(context_list)

    user_message = qna_user_message_template.replace("{context}", context_for_query)
    user_message = user_message.replace("{question}", user_input)

    # Generate the response
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": qna_system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        # gpt-oss is a reasoning model: content can be empty if reasoning used up max_tokens
        response = (response.choices[0].message.content or "").strip()
        if not response:
            response = "The model returned no answer (it may have run out of tokens while reasoning). Please try again."
    except Exception as e:
        response = f"Sorry, I encountered the following error: \n {e}"

    return response


# Streamlit App
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

retriever = None
if uploaded_files:
    st.info("Processing uploaded PDFs...")
    try:
        retriever = load_and_process_pdfs(uploaded_files)
        st.success("PDFs processed and ready for questioning!")
    except Exception as e:                                                      # Streamlit Cloud redacts uncaught errors, so show the real reason
        st.error(f"Could not process the PDFs ({type(e).__name__}): {e}")


if retriever:
    user_question = st.text_input("Ask a question about the uploaded documents:")
    if user_question:
        with st.spinner("Generating response..."):
            rag_response = generate_rag_response(user_question, retriever)
            st.markdown(rag_response)
