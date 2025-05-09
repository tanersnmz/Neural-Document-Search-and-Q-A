# app.py

import os
import torch
import numpy as np
import pandas as pd
import gradio as gr
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForQuestionAnswering,
    pipeline
)
from sklearn.neighbors import NearestNeighbors

# 1) Device
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_id = 0 if torch.cuda.is_available() else -1

# 2) Load passages list
df       = pd.read_parquet("cleaned_query_passage_pairs.parquet")
passages = df["passage"].tolist()

# 3) Load pretrained passage embeddings
passage_embeddings = np.load("passage_embeds.npy")

# 4) Build retrieval index
top_k = 5
index = NearestNeighbors(n_neighbors=top_k, metric="cosine")
index.fit(passage_embeddings)

# 5) Load retriever tokenizer & model (for query embedding)
encoder_path   = "model2/encoder"
tokenizer_path = "model2/tokenizer"
tokenizer_retriever = AutoTokenizer.from_pretrained(tokenizer_path)
retrieve_model      = AutoModel.from_pretrained(encoder_path).to(device)
retrieve_model.eval()

# 6) Load QA model & tokenizer
reader_path = "final_qa_model"
tokenizer_qa = AutoTokenizer.from_pretrained(reader_path)
model_qa     = AutoModelForQuestionAnswering.from_pretrained(reader_path).to(device)
model_qa.eval()

# 7) Build the QA pipeline
qa_pipe = pipeline(
    "question-answering",
    model=model_qa,
    tokenizer=tokenizer_qa,
    device=device_id,
    handle_impossible_answer=False,
    batch_size=top_k
)

# 8) Gradio inference function
def answer_with_context(question: str):
    # 8a) Embed the question
    with torch.no_grad():
        inputs = tokenizer_retriever(
            [question],
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        ).to(device)
        q_emb = retrieve_model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()

    # 8b) Retrieve top-k passages
    dists, idxs = index.kneighbors(q_emb, return_distance=True)
    hits = idxs[0]

    # 8c) Run the reader on those passages
    candidate_passages = [passages[i] for i in hits]
    outputs = qa_pipe(question=[question]*top_k, context=candidate_passages)

    # 8d) Pair each output with its passage and score, then sort by score desc
    scored = [
        (out["score"], candidate_passages[i], out["answer"])
        for i, out in enumerate(outputs)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    # 8e) Best answer is the first after sorting
    best_score, best_ctx, best_answer = scored[0]

    # 8f) Format the ranked passages list
    ctx_list = [
        f"{rank+1}. (score={score:.2f}) {ctx[:200]}…"
        for rank, (score, ctx, _) in enumerate(scored)
    ]

    return best_answer, "\n".join(ctx_list)

# 9) Launch the Gradio app
demo = gr.Interface(
    fn=answer_with_context,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs=[gr.Textbox(label="Answer"), gr.Textbox(label="Top-5 Retrieved Passages")],
    title="Neural Document Q&A Demo",
    description="Type a question and see the retrieved context and extracted answer."
)

if __name__ == "__main__":
    demo.launch()
