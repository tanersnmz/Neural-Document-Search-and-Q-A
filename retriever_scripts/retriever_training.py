import pandas as pd
import re
import nltk
from tqdm import tqdm
from datasets import load_dataset


print(nltk.__version__)
nltk.download('punkt_tab')

def clean_text(text):
    if text is None:
        return ""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_passages(text, min_tokens=80, max_tokens=300):
    sentences = nltk.sent_tokenize(text)
    passages = []
    current = ""
    for sent in sentences:
        if len(current.split()) + len(sent.split()) <= max_tokens:
            current += " " + sent
        else:
            if len(current.split()) >= min_tokens:
                passages.append(current.strip())
            current = sent
    if len(current.split()) >= min_tokens:
        passages.append(current.strip())
    return passages

print("Loading MS MARCO v2.1 train split")
marco = load_dataset("ms_marco", "v2.1", split="train")
query_list, passage_list = [], []
print("Building MS MARCO (query, passage) pairs…")
for item in tqdm(marco, desc="MS MARCO"):
    q = item.get("query", "")
    passages_info = item.get("passages", {})
    for is_sel, txt in zip(passages_info.get("is_selected", []),
                           passages_info.get("passage_text", [])):
        if is_sel == 1:
            cleaned = clean_text(txt)
            for p in split_passages(cleaned):
                query_list.append(q)
                passage_list.append(p)

print("Loading SQuAD train split")
squad_train = load_dataset("squad", split="train")
print("Building SQuAD (query, passage) pairs…")
for example in tqdm(squad_train, desc="SQuAD"):
    question = clean_text(example["question"])
    context  = clean_text(example["context"])
    for p in split_passages(context):  
        query_list.append(question)
        passage_list.append(p)

print("Loading HotpotQA fullwiki")
hotpot_val = load_dataset("hotpot_qa", "fullwiki", split="train")
print("Building HotpotQA (query, passage) pairs…")
for example in tqdm(hotpot_val, desc="HotpotQA"):
    q = example["question"]
    for title, sentences in zip(example["context"]["title"],
                                example["context"]["sentences"]):
        paragraph = " ".join(sentences).strip()
        cleaned   = clean_text(paragraph)
        for p in split_passages(cleaned):
            query_list.append(q)
            passage_list.append(p)


print(f"Total pairs: {len(query_list)}")
df = pd.DataFrame({"query": query_list, "passage": passage_list})
df.to_parquet("cleaned_query_passage_pairs_msmarco_hotpot.parquet")
print("Saved combined (query, passage) pairs to cleaned_query_passage_pairs_msmarco_hotpot.parquet")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.amp import autocast, GradScaler
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


model_name  = "distilbert-base-uncased"
batch_size  = 64
epochs      = 5
lr          = 2e-5
max_length  = 256



device     = torch.device("cuda")
device_ids = [0, 1]

scaler     = GradScaler()
df         = pd.read_parquet("cleaned_query_passage_pairs_msmarco_hotpot.parquet")
queries    = df["query"].tolist()
passages   = df["passage"].tolist()

class QueryPassageDataset(Dataset):
    def __init__(self, queries, passages, tokenizer, max_length=256):
        self.queries    = queries
        self.passages   = passages
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        q, p = self.queries[idx], self.passages[idx]
        q_enc = self.tokenizer(q, truncation=True, padding="max_length",
                               max_length=self.max_length, return_tensors="pt")
        p_enc = self.tokenizer(p, truncation=True, padding="max_length",
                               max_length=self.max_length, return_tensors="pt")
        return {
            "q_ids":  q_enc["input_ids"].squeeze(),
            "q_mask": q_enc["attention_mask"].squeeze(),
            "p_ids":  p_enc["input_ids"].squeeze(),
            "p_mask": p_enc["attention_mask"].squeeze(),
        }

tokenizer     = AutoTokenizer.from_pretrained(model_name)
train_dataset = QueryPassageDataset(queries, passages, tokenizer, max_length)
train_loader = DataLoader(
  train_dataset,
  batch_size=batch_size,
  shuffle=True,
  drop_last=True,
  num_workers=4,
  pin_memory=True,
  persistent_workers=True,
)
total_steps = len(train_loader) * epochs
power = 0.5
def polynomial_decay(step):
    return (1 - step / total_steps) ** power



class BiEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def encode(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        return out.last_hidden_state.mean(dim=1)

    def forward(self, q_ids, q_mask, p_ids, p_mask):
        q_emb = self.encode(q_ids, q_mask)
        p_emb = self.encode(p_ids, p_mask)
        return q_emb, p_emb


base_model = BiEncoder(model_name).to(device)
dp_model = nn.DataParallel(base_model, device_ids=device_ids)
model = torch.compile(dp_model)


optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=polynomial_decay)
loss_fn   = nn.CrossEntropyLoss()
for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        optimizer.zero_grad()
        q_ids  = batch["q_ids"].to(device)
        q_mask = batch["q_mask"].to(device)
        p_ids  = batch["p_ids"].to(device)
        p_mask = batch["p_mask"].to(device)

        with autocast(device_type="cuda"):
            q_emb, p_emb = model(q_ids, q_mask, p_ids, p_mask)
            sim_matrix   = torch.matmul(q_emb, p_emb.T)
            labels       = torch.arange(sim_matrix.size(0), device=device)
            loss         = loss_fn(sim_matrix, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")


save_dir = "/insomnia001/home/qc2354/test/model3"
try:
    model.module.encoder.save_pretrained(f"{save_dir}/encoder")
except:
    model.save_pretrained(f"{save_dir}/encoder")
tokenizer.save_pretrained(f"{save_dir}/tokenizer")

print(f"Model saved to {save_dir}")
