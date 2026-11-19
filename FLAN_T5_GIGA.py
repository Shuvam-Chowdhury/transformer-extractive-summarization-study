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
from torch.utils.data import DataLoader
from torch.optim import AdamW

import nltk
nltk.download("punkt_tab", quiet=True)   # pretrained sentence splitter (the same one the eval notebooks used)
nltk.download("wordnet", quiet=True)     # word list METEOR uses to match synonyms
nltk.download("omw-1.4", quiet=True)     # extra WordNet data METEOR can need
from nltk.tokenize import sent_tokenize
from nltk.translate.meteor_score import single_meteor_score

from transformers import T5Tokenizer, T5EncoderModel
from datasets import load_dataset
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tqdm import tqdm
from eval_utils import compute_rougeL_on_validation, evaluate_model  # shared evaluation (same for every model)

print = functools.partial(print, flush=True)

# === Settings: paths (change OUTPUT_DIR to run somewhere other than Kaggle) ===
MODEL_NAME = "google/flan-t5-small"         # Hugging Face model to download
OUTPUT_DIR = "/kaggle/working/flan_t5_giga" # Kaggle keeps this folder when you save a version
QUICK_RUN = "--quick" in sys.argv                # run "python <script>.py --quick" for a fast smoke test
NUM_ARTICLES = 200 if QUICK_RUN else 100000      # training articles to preprocess
NUM_EPOCHS = 1 if QUICK_RUN else 3               # training epochs
NUM_VAL = 50 if QUICK_RUN else 2000              # validation articles scored after each epoch
NUM_TEST = 50 if QUICK_RUN else 10000            # test articles in the final evaluation
if QUICK_RUN:
    OUTPUT_DIR = OUTPUT_DIR + "_quick"           # keep smoke-test files apart from the real run
os.makedirs(OUTPUT_DIR, exist_ok=True)        # create the folder if it doesn't exist yet

SEED = 42                         # same seed in every script so runs can be repeated
random.seed(SEED)                 # Python's shuffle (which 100k sentences are picked)
np.random.seed(SEED)              # NumPy randomness
torch.manual_seed(SEED)           # classifier-layer starting weights and DataLoader shuffling
torch.cuda.manual_seed_all(SEED)  # the same on every GPU

cache_dir = None                                                     # None = Hugging Face's default download folder
save_path_100k = os.path.join(OUTPUT_DIR, "flan_t5_giga_preprocessed.pt") # all labelled sentences
pretrained_model_path = MODEL_NAME                                   # model name passed to from_pretrained
save_dir = OUTPUT_DIR                                                # checkpoints are saved here
TEXT_KEY = "document"     # dataset column with the article text
SUMMARY_KEY = "summary"   # dataset column with the reference summary


# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Tokenizer and Sentence Splitter
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)

def tokenize_sentences(sentences):
    return tokenizer(sentences, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

def split_into_sentences(text):
    return sent_tokenize(text)

# Model Definition
class T5ForExtractiveSummarization(nn.Module):
    def __init__(self, pretrained_model_path):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(pretrained_model_path)
        self.classifier = nn.Linear(self.encoder.config.d_model, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Take [CLS]-like token
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

    dataset = load_dataset("gigaword", split="train")
    processed_samples = []
    for idx, example in enumerate(tqdm(dataset, total=NUM_ARTICLES)):
        if idx >= NUM_ARTICLES:
            break
        article = example["document"]
        summary = example["summary"]
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

    os.makedirs(os.path.dirname(save_path_100k), exist_ok=True)
    torch.save(processed_samples, save_path_100k)
    print(f"Saved {len(processed_samples)} samples to {save_path_100k}")

# Train/Val/Test Split
# === Train/Val/Test Split for GIGAWORD ===
samples = torch.load(save_path_100k, map_location="cpu")
random.shuffle(samples)

train_samples = samples[:100000]  # keep exactly 100k sentences (same for every model)
train_loader = DataLoader(SummarizationDataset(train_samples), batch_size=64, shuffle=True)

# Load validation split and create non-overlapping val/test sets
full_val_split = load_dataset("gigaword", split="validation")

val_dataset = full_val_split.select(range(2000))         # First 2,000 for validation
test_dataset = full_val_split.select(range(2000, 12000)) # Next 10,000 for testing


# Model Initialization
model = T5ForExtractiveSummarization(pretrained_model_path=pretrained_model_path).to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
    model = nn.DataParallel(model)

optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.BCEWithLogitsLoss()

save_path = os.path.join(save_dir, "best_flan_t5_gigaword_model.pt")
checkpoint_path = os.path.join(save_dir, "best_flan_t5_gigaword_checkpoint.pt")


# Training Loop
best_rougeL, best_loss = 0, float("inf")
patience, no_improve, num_epochs = 3, 0, NUM_EPOCHS

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
    val_rougeL = compute_rougeL_on_validation(model, val_dataset, tokenizer, device, TEXT_KEY, SUMMARY_KEY, max_samples=NUM_VAL)
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


# Load and Evaluate Best Model
model = T5ForExtractiveSummarization(pretrained_model_path).to(device)
state_dict = torch.load(save_path, map_location=device)
if any(k.startswith("module.") for k in state_dict.keys()):
    print(" Stripping 'module.' prefixes from DataParallel model...")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

evaluate_model(model, test_dataset, tokenizer, device, TEXT_KEY, SUMMARY_KEY, output_dir=OUTPUT_DIR, max_samples=NUM_TEST)

print("\n All done training and evaluating FLAN-T5-Small on GIGAWORD!")

