from transformers import GPT2Tokenizer, GPT2Model
import sys, os, torch, pickle, random, functools
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from datasets import load_dataset
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.meteor_score import single_meteor_score
import nltk
import torch
import torch.nn as nn
import numpy as np
import pickle
import random
import functools
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tqdm import tqdm

import nltk
nltk.download("punkt_tab", quiet=True)   # pretrained sentence splitter (the same one the eval notebooks used)
nltk.download("wordnet", quiet=True)     # word list METEOR uses to match synonyms
nltk.download("omw-1.4", quiet=True)     # extra WordNet data METEOR can need
from nltk.tokenize import sent_tokenize
from nltk.translate.meteor_score import single_meteor_score

print = functools.partial(print, flush=True)

# === Settings: paths (change OUTPUT_DIR to run somewhere other than Kaggle) ===
MODEL_NAME = "gpt2"                         # Hugging Face model to download
OUTPUT_DIR = "/kaggle/working/gpt2_cnn"     # Kaggle keeps this folder when you save a version
os.makedirs(OUTPUT_DIR, exist_ok=True)        # create the folder if it doesn't exist yet

SEED = 42                         # same seed in every script so runs can be repeated
random.seed(SEED)                 # Python's shuffle (which 100k sentences are picked)
np.random.seed(SEED)              # NumPy randomness
torch.manual_seed(SEED)           # classifier-layer starting weights and DataLoader shuffling
torch.cuda.manual_seed_all(SEED)  # the same on every GPU

cache_dir = None                                                     # None = Hugging Face's default download folder
sample_ckpt_path = os.path.join(OUTPUT_DIR, "gpt2_cnn_preprocess_progress.pt")  # partial progress while preprocessing
save_path_100k = os.path.join(OUTPUT_DIR, "gpt2_cnn_preprocessed.pt")  # all labelled sentences
save_dir = OUTPUT_DIR                                                # checkpoints are saved here


# Load sentence tokenizer

def split_into_sentences(text):
    return sent_tokenize(text)

def label_sentences(sentences, reference, top_k=3):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(reference, sent)['rougeL'].fmeasure for sent in sentences]
    top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [1 if i in top_idxs else 0 for i in range(len(sentences))]

# Tokenizer and device setup
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
tokenizer.pad_token = tokenizer.eos_token
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GPT-2 Model Definition
class GPT2ExtractiveSummarizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = GPT2Model.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_rep).squeeze(-1)
        return logits

# Dataset Wrapper
class ExtractiveDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "label": torch.tensor(sample["label"], dtype=torch.float)
        }

# Load CNN/DailyMail dataset
raw_dataset = load_dataset("cnn_dailymail", "3.0.0")
train_articles = raw_dataset["train"]
val_data = raw_dataset["validation"].select(range(2000))
test_data = raw_dataset["test"].select(range(10000))

# Preprocessing (skipped if the preprocessed file is already saved)
if os.path.exists(save_path_100k):
    print(f"Found preprocessed samples at {save_path_100k}")
    samples = torch.load(save_path_100k, map_location="cpu")
else:
    samples = []
    print(" Preprocessing 100k training samples with checkpointing...")
    for i in range(min(100000, len(train_articles))):
        article = train_articles[i]['article']
        summary = train_articles[i]['highlights']
        sentences = split_into_sentences(article)
        if not sentences:
            continue
        labels = label_sentences(sentences, summary)
        tokenized = tokenizer(sentences, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        for j in range(len(sentences)):
            samples.append({
                "input_ids": tokenized['input_ids'][j],
                "attention_mask": tokenized['attention_mask'][j],
                "label": labels[j]
            })
        if (i + 1) % 10000 == 0:
            print(f" Processed {i+1} articles — saving progress")
            torch.save(samples, sample_ckpt_path)

    torch.save(samples, save_path_100k)

# Training setup: shuffle and keep 100k sentences (same for every model)
random.shuffle(samples)
samples = samples[:100000]
train_dataset = ExtractiveDataset(samples)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = GPT2ExtractiveSummarizer().to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
    model = nn.DataParallel(model)

optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.BCEWithLogitsLoss()

save_path = os.path.join(save_dir, "best_gpt2_extractive.pt")
checkpoint_path = os.path.join(save_dir, "gpt2_extractive_checkpoint.pt")

start_epoch = 0
best_rougel = 0.0
best_loss = float("inf")                    # training loss of the best epoch (breaks ties in ROUGE-L)
patience, no_improve, num_epochs = 3, 0, 3   # same early-stopping settings as the other scripts

if os.path.exists(checkpoint_path):
    print(" Resuming from checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_rougel = checkpoint['best_rougel']
    best_loss = checkpoint.get('best_loss', float("inf"))   # .get() so older checkpoints still load
    no_improve = checkpoint.get('no_improve', 0)
    start_epoch = checkpoint['epoch'] + 1

# Evaluation Function
def evaluate_rougel(model, val_data, tokenizer, device, max_samples=2000):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    total_score = 0.0
    with torch.no_grad():
        for i in range(min(max_samples, len(val_data))):
            article = val_data[i]['article']
            reference = val_data[i]['highlights']
            sentences = split_into_sentences(article)
            if not sentences:
                continue
            tokenized = tokenizer(sentences, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)
            logits = model(input_ids, attention_mask)
            topk = torch.topk(logits, k=min(3, len(sentences))).indices.tolist()
            pred_summary = " ".join([sentences[i] for i in topk])
            score = scorer.score(reference, pred_summary)['rougeL'].fmeasure
            total_score += score
    return total_score / max_samples

# Training Loop
print("🚀 Training GPT2 Extractive Model")
for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"[Epoch {epoch+1}] Batch {batch_idx+1} Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f" Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

    val_rougel = evaluate_rougel(model, val_data, tokenizer, device)
    print(f" Validation ROUGE-L: {val_rougel:.4f}")

    # Same rule as the other scripts: higher validation ROUGE-L wins, lower training loss breaks a tie
    is_better = val_rougel > best_rougel or (val_rougel == best_rougel and avg_loss < best_loss)
    if is_better:
        best_rougel, best_loss = val_rougel, avg_loss
        torch.save(model.state_dict(), save_path)
        print(f" New best model saved (ROUGE-L {val_rougel:.4f})")
        no_improve = 0
    else:
        no_improve += 1
        print(f" No improvement | Patience: {no_improve}/{patience}")
        if no_improve >= patience:
            print(" Early stopping.")
            break

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_rougel': best_rougel,
        'best_loss': best_loss,
        'no_improve': no_improve
    }, checkpoint_path)
    print(f" Checkpoint saved at epoch {epoch+1}")

# Final Evaluation on Test Set
# Load the best model into a fresh single-GPU copy
model = GPT2ExtractiveSummarizer().to(device)
state_dict = torch.load(save_path, map_location=device)
if any(k.startswith("module.") for k in state_dict.keys()):
    print(" Stripping 'module.' prefixes from DataParallel model...")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

def evaluate_on_test(model, dataset, tokenizer, device, max_samples=10000):
    model.eval()
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    meteor_total, rouge1_total, rouge2_total, rougeL_total = 0, 0, 0, 0
    references, predictions = [], []

    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            article = dataset[i]['article']
            reference = dataset[i]['highlights']
            sentences = split_into_sentences(article)
            if not sentences:
                continue
            tokenized = tokenizer(sentences, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)
            logits = model(input_ids, attention_mask)
            topk = torch.topk(logits, k=min(3, len(sentences))).indices.tolist()
            pred_summary = " ".join([sentences[i] for i in topk])
            scores = rouge.score(reference, pred_summary)
            rouge1_total += scores['rouge1'].fmeasure
            rouge2_total += scores['rouge2'].fmeasure
            rougeL_total += scores['rougeL'].fmeasure
            meteor_total += single_meteor_score(reference.split(), pred_summary.split())
            references.append(reference)
            predictions.append(pred_summary)

    n = len(predictions)
    print(f"\n Final Evaluation on {n} test samples")
    print(f"ROUGE-1 F1: {rouge1_total / n:.4f}")
    print(f"ROUGE-2 F1: {rouge2_total / n:.4f}")
    print(f"ROUGE-L F1: {rougeL_total / n:.4f}")
    print(f"METEOR:     {meteor_total / n:.4f}")
    precision, recall, f1 = bert_score(predictions, references, lang='en', verbose=False)
    print(f"BERTScore P/R/F1: {precision.mean().item():.4f} / {recall.mean().item():.4f} / {f1.mean().item():.4f}")

# Run test evaluation
evaluate_on_test(model, test_data, tokenizer, device)
