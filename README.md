# Neural Document Search & Question‐Answering Demo

End-to-end two‐stage pipeline that retrieves relevant passages and extracts precise answers.

---

## 🧪 Experimental Setup

- Python version: 3.9.9
- PyTorch version: 2.3.1+cu121
- CUDA available: True
- CUDA version: 12.1
- GPU model: Tesla T4
- Number of GPUs: 1
- Available GPU memory: 15.64 GB
- Using device: cuda

---

## 📖 Project Overview

1. **Retriever**: DistilBERT bi-encoder trained on MS MARCO v2.1 + SQuAD + HotpotQA pairs to embed queries and passages and return top-K documents.  
2. **Reader**: DistilBERT span-prediction head fine-tuned on SQuAD v1.1 to extract exact answer spans from the retrieved passages.  
3. **Demo**: A Gradio app for live querying and visualization of top-5 supporting passages.

---

## 📁 Directory Layout

```plaintext
.
├── retriever_scripts/
│   ├── retriever_training.py       # build corpus & train bi-encoder
│   └── SQUAD_retriever_eval.ipynb  # retriever evaluation notebook
│
├── reader.ipynb                    # fine-tune reader, eval, end-to-end pipeline (after getting the retriever files, you can use this file to test entire pipeline)
│
├── model2/                         # retriever model files (in drive, see below link)
├── final_qa_model/                 # reader model files (in drive, see below link)
├── model_repository/               # contains NVDIA Triton deployment files
│
├── cleaned_query_passage_pairs.parquet # in drive, see below link
├── passage_embeds.npy              # precomputed passage embeddings, in drive, see below link
│
├── app.py                # live demo interface
├── requirements.txt                # Python dependencies
└── README.md
```

## Setup

Install the required packages:
`pip install -r requirements.txt`

## Retriever

### Train the retriever
`python retriever_scripts/retriever_training.py`

### Evaluate the retriever
Open the notebook: `retriever_scripts/SQUAD_retriever_eval.ipynb`

## Reader
Open and run `reader.ipynb` to train and evaluate the reader. This file also utilizes entire pipeline using reader + retriever.

## Quick Demo (Runnig in locally, refer the drive link below)
### 1. Create a virtual environment
python -m venv venv && source venv/bin/activate

### 2. Install dependencies
pip install -r requirements.txt   

### 3. Launch Gradio App
python app.py

This opens app in your local machine with specific url.  
Type a question (e.g. “What causes rainbows to form?”) and press Submit;
the app shows the extracted answer and the top-5 supporting passages.

## Results

Our retriever model achieves a Recall\@5 of **49.9 %** and a Recall\@20 of **74.4 %**. In comparison, the current state-of-the-art reported by Chen et al. (2023) reaches **63.3 %** and **75.0 %** for Recall\@5 and Recall\@20 on the same benchmark.
For the reader, our fine-tuned model obtains **78.6 %** Exact Match and **86.5 %** F1 on the SQuAD v1.1 validation set, while the best published results from Jun et al. (2022) are **90.6 %** Exact Match and **95.7 %** F1.
For details, check the final report.


## Notes
- All trained retriever and reader model files can be downloaded from this link, for DEMO download the files in the link and refer the quick demo instructions above (it requires Columbia Email): https://drive.google.com/drive/folders/1W7k1mfacTO3114hez1BsbBRAXyZoOk2r?usp=sharing
- Use a GPU (e.g., `cuda:0`) for embedding and inference.
- Ensure that retrieved passages align correctly with the reader input.

## License

MIT License

