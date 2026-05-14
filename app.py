import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Bloom · Document Intelligence",
    page_icon="🌸",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Palette ──────────────────────────────── */
:root {
    --blush:       #f9c8c8;
    --peach:       #ffb996;
    --peach-soft:  #ffd6be;
    --rose:        #e8748a;
    --rose-deep:   #c45470;
    --cream:       #fff8f5;
    --petal:       #fef0ec;
    --fog:         #fde8e3;
    --text-dark:   #2d1a1a;
    --text-mid:    #7a4a50;
    --text-light:  #b07a82;
    --border:      #f5cec8;
    --shadow:      rgba(196,84,112,.13);
}

/* ── Reset ────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--cream);
    color: var(--text-dark);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 2.5rem 5rem;
    max-width: 760px;
}

/* ── Hero banner ──────────────────────────── */
.hero {
    background: linear-gradient(135deg, var(--blush) 0%, var(--peach-soft) 50%, var(--fog) 100%);
    border-radius: 0 0 32px 32px;
    padding: 3rem 2.5rem 2.6rem;
    text-align: center;
    margin: 0 -2.5rem 2.8rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(circle at 15% 80%, rgba(255,185,150,.45) 0%, transparent 55%),
        radial-gradient(circle at 85% 20%, rgba(249,200,200,.55) 0%, transparent 55%);
    pointer-events: none;
}
.hero-tag {
    display: inline-block;
    background: var(--rose);
    color: #fff;
    font-size: .68rem;
    font-weight: 500;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    padding: .25rem .75rem;
    border-radius: 99px;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: 3rem;
    font-weight: 400;
    color: var(--text-dark);
    margin: 0 0 .35rem;
    line-height: 1.1;
    position: relative;
}
.hero h1 em {
    font-style: italic;
    color: var(--rose-deep);
}
.hero .sub {
    font-size: .93rem;
    font-weight: 300;
    color: var(--text-mid);
    letter-spacing: .3px;
    position: relative;
}
.hero .bloom-icon {
    font-size: 2.2rem;
    margin-bottom: .6rem;
    display: block;
    position: relative;
}

/* ── Section label ────────────────────────── */
.label {
    font-size: .68rem;
    font-weight: 500;
    letter-spacing: 2.2px;
    text-transform: uppercase;
    color: var(--rose);
    margin-bottom: .55rem;
}

/* ── Upload zone ──────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
    background: var(--petal) !important;
    border: 2px dashed var(--peach) !important;
    border-radius: 16px !important;
    transition: border-color .2s, background .2s;
}
[data-testid="stFileUploaderDropzone"]:hover {
    background: var(--fog) !important;
    border-color: var(--rose) !important;
}
/* Force all text inside uploader to be dark/visible */
[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploader"] *,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small {
    color: var(--text-dark) !important;
}
[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-mid) !important;
    font-size: .85rem !important;
}

/* ── Buttons ──────────────────────────────── */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: .88rem !important;
    font-weight: 500 !important;
    letter-spacing: .4px !important;
    color: #fff !important;
    background: linear-gradient(135deg, var(--rose) 0%, var(--rose-deep) 100%) !important;
    border: none !important;
    border-radius: 99px !important;
    padding: .6rem 1.8rem !important;
    box-shadow: 0 4px 16px var(--shadow) !important;
    transition: transform .18s, box-shadow .18s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px var(--shadow) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Text input ───────────────────────────── */
.stTextInput > div > div > input {
    font-family: 'DM Sans', sans-serif !important;
    font-size: .95rem !important;
    font-weight: 300 !important;
    background: var(--petal) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-dark) !important;
    padding: .65rem 1rem !important;
    transition: border-color .2s, box-shadow .2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--rose) !important;
    box-shadow: 0 0 0 3px rgba(228,116,138,.15) !important;
}
.stTextInput label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: .85rem !important;
    font-weight: 400 !important;
    color: var(--text-mid) !important;
}

/* ── Alerts ───────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: .88rem !important;
}

/* ── Answer card ──────────────────────────── */
.answer-wrap {
    margin-top: 1.4rem;
    background: linear-gradient(135deg, var(--fog) 0%, var(--petal) 100%);
    border: 1.5px solid var(--border);
    border-radius: 20px;
    padding: 1.6rem 1.8rem;
    box-shadow: 0 6px 24px var(--shadow);
    position: relative;
    overflow: hidden;
}
.answer-wrap::before {
    content: '';
    position: absolute;
    top: -20px; right: -20px;
    width: 100px; height: 100px;
    background: radial-gradient(circle, var(--peach-soft) 0%, transparent 70%);
    pointer-events: none;
}
.answer-wrap .a-tag {
    font-size: .64rem;
    font-weight: 500;
    letter-spacing: 2.2px;
    text-transform: uppercase;
    color: var(--rose);
    margin-bottom: .7rem;
    display: flex;
    align-items: center;
    gap: .4rem;
}
.answer-wrap .a-tag::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}
.answer-wrap .a-body {
    font-size: .97rem;
    font-weight: 300;
    line-height: 1.8;
    color: var(--text-dark);
}

/* ── Divider ──────────────────────────────── */
.divider {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 2rem 0;
    color: var(--text-light);
    font-size: .8rem;
    letter-spacing: 1px;
}
.divider::before, .divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Spinner ──────────────────────────────── */
[data-testid="stSpinner"] p {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-mid) !important;
    font-style: italic;
    font-weight: 300;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="bloom-icon">🌸</span>
    <div class="hero-tag">RAG · Retrieval Augmented Generation</div>
    <h1>RAG <em>Bloom</em></h1>
    <p class="sub">Drop a document. Ask anything. Watch knowledge blossom.</p>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown('<div class="label">Upload document</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop your PDF here or click to browse",
    type="pdf",
    label_visibility="collapsed",
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success(f"🌸 **{uploaded_file.name}** uploaded successfully!")

    if st.button("✦  Index this Document"):
        with st.spinner("Reading and indexing your document…"):

            loader = PyPDFLoader(file_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            chunks = splitter.split_documents(docs)

            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
            )

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory="chroma_db",
            )
            vectorstore.persist()

        st.success("✦ Index is ready — start asking questions below!")

# ── Q&A ───────────────────────────────────────────────────────────────────────
if os.path.exists("chroma_db"):

    st.markdown('<div class="divider">✦ ask ✦</div>', unsafe_allow_html=True)
    st.markdown('<div class="label">Your question</div>', unsafe_allow_html=True)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model,
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )

    llm = ChatMistralAI(model="mistral-small-2506")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful AI assistant.
Use ONLY the provided context to answer the question.
If the answer is not present in the context, say: "I could not find the answer in the document." """,
        ),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
    ])

    query = st.text_input(
        "What would you like to know?",
        placeholder="e.g. Summarise the key arguments in chapter 2…",
    )

    if query:
        with st.spinner("Blooming an answer for you…"):
            docs = retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            final_prompt = prompt.invoke({"context": context, "question": query})
            response = llm.invoke(final_prompt)

        st.markdown(f"""
        <div class="answer-wrap">
            <div class="a-tag">🌸 &nbsp;Bloom Answer</div>
            <div class="a-body">{response.content}</div>
        </div>
        """, unsafe_allow_html=True)
