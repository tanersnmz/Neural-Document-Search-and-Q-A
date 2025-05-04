# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering, pipeline
from sklearn.neighbors import NearestNeighbors

@st.cache_resource
def load_models():
    # 1) Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2) Passages + index
    df = pd.read_parquet("cleaned_query_passage_pairs.parquet")
    passages = df["passage"].tolist()
    emb = np.load("passage_embeds.npy")
    index = NearestNeighbors(n_neighbors=5, metric="cosine").fit(emb)
    # 3) Retriever
    tok_ret = AutoTokenizer.from_pretrained("model2/tokenizer")
    mdl_ret = AutoModel.from_pretrained("model2/encoder").to(device).eval()
    # 4) Reader
    reader_path = "final_qa_model"
    tok_qa = AutoTokenizer.from_pretrained(reader_path)
    mdl_qa = AutoModelForQuestionAnswering.from_pretrained(reader_path).to(device).eval()
    qa_pipe = pipeline(
        "question-answering", model=mdl_qa, tokenizer=tok_qa,
        device=0 if torch.cuda.is_available() else -1,
        handle_impossible_answer=False, batch_size=5
    )
    return device, passages, index, tok_ret, mdl_ret, qa_pipe

device, passages, index, tok_ret, mdl_ret, qa_pipe = load_models()

st.title("Neural Document Q&A Demo")
question = st.text_input("Enter your question:")

if question:
    # embed + retrieve
    inputs = tok_ret([question], truncation=True, padding="max_length",
                     max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        q_emb = mdl_ret(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
    dists, idxs = index.kneighbors(q_emb)
    hits = idxs[0]

    # read + rank
    candidates = [passages[i] for i in hits]
    outputs    = qa_pipe(question=[question]*5, context=candidates)
    scored     = sorted(zip(outputs, candidates),
                        key=lambda x: x[0]["score"], reverse=True)

    best, best_ctx = scored[0]
    st.subheader("Answer")
    st.write(best["answer"])

    st.subheader("Top-5 Contexts")
    for rank, (out, ctx) in enumerate(scored, start=1):
        st.write(f"**{rank}.** (score={out['score']:.2f}) {ctx[:200]}…")
