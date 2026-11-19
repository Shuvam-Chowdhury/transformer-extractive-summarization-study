#!/usr/bin/env python
# coding: utf-8

import sys

import os
os.environ['PYTHONUNBUFFERED'] = "1"

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

import nltk
nltk.download("punkt_tab", quiet=True)   # pretrained sentence splitter (the same one the eval notebooks used)
nltk.download("wordnet", quiet=True)     # word list METEOR uses to match synonyms
nltk.download("omw-1.4", quiet=True)     # extra WordNet data METEOR can need
from nltk.tokenize import sent_tokenize
from nltk.translate.meteor_score import single_meteor_score

from transformers import BartTokenizer, BartModel
from datasets import load_dataset
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tqdm import tqdm

print = functools.partial(print, flush=True)

# === Settings: paths (change OUTPUT_DIR to run somewhere other than Kaggle) ===
MODEL_NAME = "facebook/bart-base"           # Hugging Face model to download
OUTPUT_DIR = "/kaggle/working/bart_cnn"     # Kaggle keeps this folder when you save a version
os.makedirs(OUTPUT_DIR, exist_ok=True)        # create the folder if it doesn't exist yet

SEED = 42                         # same seed in every script so runs can be repeated
random.seed(SEED)                 # Python's shuffle (which 100k sentences are picked)
np.random.seed(SEED)              # NumPy randomness
torch.manual_seed(SEED)           # classifier-layer starting weights and DataLoader shuffling
torch.cuda.manual_seed_all(SEED)  # the same on every GPU

cache_dir = None                                                     # None = Hugging Face's default download folder
save_path_100k = os.path.join(OUTPUT_DIR, "bart_cnn_preprocessed.pt") # all labelled sentences
pretrained_model_path = MODEL_NAME                                   # model name passed to from_pretrained
save_dir = OUTPUT_DIR                                                # checkpoints are saved here


# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Tokenizer and Sentence Splitter
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)



def tokenize_sentences(sentences):
    return tokenizer(sentences, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

def split_into_sentences(text):
    return sent_tokenize(text)

# Model Definition
class BartForExtractiveSummarization(nn.Module):
    def __init__(self, pretrained_model_path):
        super().__init__()
        self.bart = BartModel.from_pretrained(pretrained_model_path)
        self.classifier = nn.Linear(self.bart.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bart.encoder(input_ids=input_ids, attention_mask=attention_mask)  # encoder only, as in the report
        cls_output = outputs.last_hidden_state[:, 0, :]  # encoder's first token (<s>)
        logits = self.classifier(cls_output).squeeze(-1)
        return logits

# Dataset Wrapper
class SummarizationDataset(torch.utils.data.Dataset):
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

# Preprocessing

if os.path.exists(save_path_100k):
    print(f"Found preprocessed samples at {save_path_100k}")
else:
    print("Preprocessing not found. Starting preprocessing...")

    def label_sentences(sentences, reference, top_k=3):
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = [scorer.score(reference, sent)['rougeL'].fmeasure for sent in sentences]
        top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [1 if i in top_idxs else 0 for i in range(len(sentences))]

    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")
    processed_samples = []
    max_articles = 100000

    for idx, example in enumerate(tqdm(dataset, total=max_articles, desc="Preprocessing")):
        if idx >= max_articles:
            break
        article = example["article"]
        summary = example["highlights"]

        sentences = split_into_sentences(article)
        if not sentences:
            continue
        labels = label_sentences(sentences, summary)
        for j, sentence in enumerate(sentences):
            prompt_tokens = tokenizer(sentence, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            processed_samples.append({
                "input_ids": prompt_tokens['input_ids'].squeeze(0),
                "attention_mask": prompt_tokens['attention_mask'].squeeze(0),
                "label": labels[j]
            })

        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx+1} articles - total samples so far: {len(processed_samples)}")

    os.makedirs(os.path.dirname(save_path_100k), exist_ok=True)
    torch.save(processed_samples, save_path_100k)
    print(f" Saved {len(processed_samples)} samples to {save_path_100k}")

# Train/Val Split
samples = torch.load(save_path_100k, map_location="cpu")
random.shuffle(samples)
samples = samples[:100000]

train_samples = samples  # all 100k sentences (same for every model)

train_loader = DataLoader(SummarizationDataset(train_samples), batch_size=64, shuffle=True)

val_dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation[:2000]")
test_dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:10000]")

# Model Initialization

model = BartForExtractiveSummarization(pretrained_model_path=pretrained_model_path).to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
    model = nn.DataParallel(model)

# Optimizer and Loss
learning_rate = 2e-5
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
loss_fn = nn.BCEWithLogitsLoss()

save_path = os.path.join(save_dir, "best_bart_cnn_dailymail_model.pt")
checkpoint_path = os.path.join(save_dir, "best_bart_cnn_dailymail_checkpoint.pt")

# Helper for ROUGE-L Computation
def compute_rougeL_on_validation(model, val_dataset, tokenizer, device, max_samples=2000):
    model.eval()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    total_rougeL = 0
    n = 0
    with torch.no_grad():
        for i in range(min(max_samples, len(val_dataset))):
            article = val_dataset[i]['article']
            summary = val_dataset[i]['highlights']
            sentences = split_into_sentences(article)
            if not sentences:
                continue
            tokenized = tokenize_sentences(sentences)
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)
            logits = model(input_ids, attention_mask).squeeze(0)
            if logits.dim() == 0:
                continue
            top_indices = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:3]
            pred_summary = " ".join([sentences[i] for i in top_indices])
            score = scorer.score(summary, pred_summary)
            total_rougeL += score["rougeL"].fmeasure
            n += 1
    return total_rougeL / n if n else 0.0

# Training Loop
best_rougeL, best_loss = 0, float("inf")
patience, no_improve, num_epochs = 3, 0, 3

print("\n Training started...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    print(f" Epoch {epoch+1}/{num_epochs}")

    for i, batch in enumerate(train_loader):
        inputs = batch["input_ids"].to(device)
        masks = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(inputs, masks)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f"Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    val_rougeL = compute_rougeL_on_validation(model, val_dataset, tokenizer, device)
    print(f" Epoch {epoch+1} Summary | Train Loss: {avg_loss:.4f} | Validation ROUGE-L: {val_rougeL:.4f}")

    is_better = val_rougeL > best_rougeL or (val_rougeL == best_rougeL and avg_loss < best_loss)
    if is_better:
        best_rougeL, best_loss = val_rougeL, avg_loss
        torch.save(model.state_dict(), save_path)
        print(" Best model saved based on Validation ROUGE-L!")
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(" Early stopping triggered (no improvement).")
            break

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_rougeL': best_rougeL,
        'best_loss': best_loss
    }, checkpoint_path)
    print(f" Checkpoint saved at epoch {epoch+1}")

# Final Evaluation
def evaluate_model(model, dataset, tokenizer, device, max_samples=10000):
    model.eval()
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    meteor_total, rouge1_total, rouge2_total, rougeL_total = 0, 0, 0, 0
    references, predictions = [], []
    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            article = dataset[i]['article']
            summary = dataset[i]['highlights']
            sentences = split_into_sentences(article)
            if not sentences:
                continue
            tokenized = tokenize_sentences(sentences)
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)
            logits = model(input_ids, attention_mask).squeeze(0)
            if logits.dim() == 0:
                continue
            top_indices = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:3]
            pred_summary = " ".join([sentences[i] for i in top_indices])
            scores = rouge.score(summary, pred_summary)
            rouge1_total += scores['rouge1'].fmeasure
            rouge2_total += scores['rouge2'].fmeasure
            rougeL_total += scores['rougeL'].fmeasure
            meteor_total += single_meteor_score(summary.split(), pred_summary.split())
            references.append(summary)
            predictions.append(pred_summary)
    precision, recall, f1 = bert_score(predictions, references, lang='en', verbose=False)
    n = len(predictions)
    print(f"\n Final Evaluation on {n} samples")
    print(f"ROUGE-1 F1: {rouge1_total / n:.4f}")
    print(f"ROUGE-2 F1: {rouge2_total / n:.4f}")
    print(f"ROUGE-L F1: {rougeL_total / n:.4f}")
    print(f"METEOR:     {meteor_total / n:.4f}")
    print(f"BERTScore P/R/F1: {precision.mean().item():.4f} / {recall.mean().item():.4f} / {f1.mean().item():.4f}")

# Load and Evaluate Best Model
model = BartForExtractiveSummarization(pretrained_model_path=pretrained_model_path).to(device)
state_dict = torch.load(save_path, map_location=device)
if any(k.startswith("module.") for k in state_dict.keys()):
    print(" Stripping 'module.' prefixes from DataParallel model...")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

evaluate_model(model, test_dataset, tokenizer, device)  # test[:10000], same as every model

print("\n All done training and evaluating BART on CNN/DailyMail!")