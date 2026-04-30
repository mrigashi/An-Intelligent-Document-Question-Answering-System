"""
PDF RAG Q&A System
------------------

Free API keys:
  • Cohere → https://dashboard.cohere.com   (free tier, no CC needed)
  • Groq   → https://console.groq.com

Run:
    streamlit run app.py
"""

import textwrap
import time
import numpy as np
import streamlit as st
import PyPDF2
import faiss
import cohere
from groq import Groq

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="An Intelligent Document Question Answering System Using Generative AI", layout="wide")

st.markdown("""
<style>
.pipeline-box {
    padding: 12px 14px; border-radius: 8px;
    border: 1px solid #e0e0e0; background: #f9f9f9;
    text-align: center; font-size: 13px; font-weight: 500;
}
.pipeline-box .num { font-family: monospace; font-size: 11px; color: #888; }
.answer-box {
    padding: 16px 18px; border-radius: 10px;
    border: 1px solid #d0e8ff; background: #f0f7ff;
    font-size: 15px; line-height: 1.75;
}
.ctx-box {
    padding: 12px 14px; border-radius: 8px;
    border: 1px solid #e5e5e5; background: #f5f5f5;
    font-family: monospace; font-size: 12px; line-height: 1.6; color: #555;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
COHERE_EMBED_MODEL = "embed-english-light-v3.0"   # 384-dim, free tier
GROQ_MODEL         = "llama-3.1-8b-instant"
CHUNK_SIZE         = 500
CHUNK_OVERLAP      = 80
TOP_K              = 4
EMBED_BATCH_SIZE   = 96   # Cohere allows up to 96 texts per call

# ─── Helper: PDF → text ───────────────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file) -> str:
    reader = PyPDF2.PdfReader(uploaded_file)
    return "\n\n".join(
        p.extract_text().strip()
        for p in reader.pages
        if p.extract_text()
    )

# ─── Helper: text → overlapping chunks ───────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks

# ─── Helper: Cohere batch embed ───────────────────────────────────────────────
def embed_texts(co: cohere.ClientV2, texts: list[str],
                input_type: str = "search_document") -> np.ndarray:
    """
    input_type = "search_document" when indexing chunks
    input_type = "search_query"    when embedding a user question
    """
    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch    = texts[i : i + EMBED_BATCH_SIZE]
        response = co.embed(
            texts=batch,
            model=COHERE_EMBED_MODEL,
            input_type=input_type,
            embedding_types=["float"],
        )
        all_embeddings.extend(response.embeddings.float_)
        if i + EMBED_BATCH_SIZE < len(texts):
            time.sleep(0.2)

    arr   = np.array(all_embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-9)   # L2-normalise for cosine via dot product

# ─── Helper: build FAISS index ────────────────────────────────────────────────
def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

# ─── Helper: semantic retrieve ────────────────────────────────────────────────
def retrieve(co: cohere.ClientV2, query: str, chunks: list[str],
             index: faiss.IndexFlatIP, k: int = TOP_K) -> list[str]:
    q_emb   = embed_texts(co, [query], input_type="search_query")
    _, idxs = index.search(q_emb, k)
    return [chunks[i] for i in idxs[0] if i < len(chunks)]

# ─── Helper: generate answer via Groq ────────────────────────────────────────
def generate_answer(query: str, context_chunks: list[str], groq_key: str) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    client  = Groq(api_key=groq_key)
    system  = textwrap.dedent("""
        You are a precise Q&A assistant. Answer using ONLY the provided context
        excerpts from the PDF. If the answer isn't there, say clearly:
        "This information is not found in the document."
        Be concise and reference the relevant part of the context.
    """).strip()
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Context:\n\n{context}\n\n---\n\nQuestion: {query}"},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return resp.choices[0].message.content

# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("An Intelligent Document Question Answering System Using Generative AI")

st.divider()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    st.markdown("**Cohere API key (free)**")
    st.markdown("[dashboard.cohere.com](https://dashboard.cohere.com)")
    cohere_key = st.text_input("Cohere API Key", type="password", key="coh")

    st.markdown("**Groq API key (free)**")
    st.markdown("[console.groq.com](https://console.groq.com)")
    groq_key = st.text_input("Groq API Key", type="password", key="grq")

    st.markdown("---")
    st.markdown("**Chunking**")
    chunk_size    = st.slider("Chunk size (chars)",       100, 800, CHUNK_SIZE,    step=50)
    chunk_overlap = st.slider("Chunk overlap (chars)",      0, 200, CHUNK_OVERLAP, step=20)
    top_k         = st.slider("Top-K chunks to retrieve",  1,   8, TOP_K)

    st.markdown("---")
    st.markdown("**Models**")
    st.code(COHERE_EMBED_MODEL, language=None)
    st.caption("384-dim semantic embeddings — free tier")
    st.code(GROQ_MODEL, language=None)
    st.caption("Generation — Groq free tier")

# ─── Main layout ──────────────────────────────────────────────────────────────
col_upload, col_qa = st.columns([1, 1.6], gap="large")

with col_upload:
    st.subheader("1. Upload PDF")
    uploaded = st.file_uploader("Drop your PDF here", type="pdf", label_visibility="collapsed")

    if uploaded:
        if not cohere_key:
            st.error("Enter your Cohere API key in the sidebar first.")
            st.stop()

        co = cohere.ClientV2(api_key=cohere_key)

        with st.spinner("Extracting text…"):
            raw_text = extract_text_from_pdf(uploaded)

        if not raw_text.strip():
            st.error("No text found — PDF may be image/scanned.")
            st.stop()

        chunks = chunk_text(raw_text, chunk_size, chunk_overlap)
        st.info(f"{len(raw_text):,} chars · {len(chunks)} chunks — embedding now…")

        progress = st.progress(0)
        all_embs = []
        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch    = chunks[i : i + EMBED_BATCH_SIZE]
            response = co.embed(
                texts=batch,
                model=COHERE_EMBED_MODEL,
                input_type="search_document",
                embedding_types=["float"],
            )
            all_embs.extend(response.embeddings.float_)
            progress.progress(min((i + EMBED_BATCH_SIZE) / len(chunks), 1.0))
            if i + EMBED_BATCH_SIZE < len(chunks):
                time.sleep(0.2)

        progress.empty()
        arr   = np.array(all_embs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr   = arr / np.maximum(norms, 1e-9)

        faiss_index = build_index(arr)
        st.success(f"✓ Indexed {len(chunks)} chunks (dim={arr.shape[1]})")

        st.session_state["chunks"]      = chunks
        st.session_state["faiss_index"] = faiss_index
        st.session_state["co"]          = co

        with st.expander("Preview extracted text"):
            st.text(raw_text[:1500] + ("…" if len(raw_text) > 1500 else ""))

with col_qa:
    st.subheader("2. Ask a Question")

    if "chunks" not in st.session_state:
        st.info("Upload a PDF on the left to get started.")
    else:
        query = st.text_input(
            "Your question",
            placeholder="e.g. What are the main findings?",
            label_visibility="collapsed",
        )

        if st.button("Ask ↗", type="primary", disabled=not query.strip()):
            if not groq_key:
                st.error("Enter your Groq API key in the sidebar.")
            else:
                prog = st.progress(0, text="Embedding query semantically…")

                retrieved = retrieve(
                    st.session_state["co"],
                    query,
                    st.session_state["chunks"],
                    st.session_state["faiss_index"],
                    k=top_k,
                )
                prog.progress(50, text="Generating answer…")

                if not retrieved:
                    prog.empty()
                    st.warning("No relevant chunks found. Try rephrasing your question.")
                else:
                    with st.expander(f"🔍 Retrieved context ({len(retrieved)} chunks)", expanded=False):
                        for i, chunk in enumerate(retrieved, 1):
                            st.markdown(
                                f'<div class="ctx-box"><b>Chunk {i}</b><br>{chunk}</div><br>',
                                unsafe_allow_html=True,
                            )
                    try:
                        answer = generate_answer(query, retrieved, groq_key)
                        prog.progress(100, text="Done ✓")
                        st.markdown("**Answer**")
                        st.markdown(
                            f'<div class="answer-box">{answer}</div>',
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        prog.empty()
                        st.error(f"Error: {e}")

        st.markdown("**Try asking:**")
        suggestions = [
            "What is the main topic?",
            "Summarise the key findings.",
            "What methodology is used?",
            "What are the conclusions?",
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            if cols[i % 2].button(s, key=f"sug_{i}"):
                st.session_state["_suggest"] = s
                st.rerun()

if "_suggest" in st.session_state:
    st.query_params["q"] = st.session_state.pop("_suggest")
