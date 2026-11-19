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
from eval_utils import compute_rougeL_on_validation, evaluate_model  # shared evaluation (same for every model)

import nltk
nltk.download("punkt_tab", quiet=True)   # pretrained sentence splitter (the same one the eval notebooks used)
nltk.download("wordnet", quiet=True)     # word list METEOR uses to match synonyms
nltk.download("omw-1.4", quiet=True)     # extra WordNet data METEOR can need
from nltk.tokenize import sent_tokenize
from nltk.translate.meteor_score import single_meteor_score

print = functools.partial(print, flush=True)

# === Settings: paths (change OUTPUT_DIR to run somewhere other than Kaggle) ===
MODEL_NAME = "gpt2"                         # Hugging Face model to download
OUTPUT_DIR = "/kaggle/working/gpt2_giga"    # Kaggle keeps this folder when you save a version
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
sample_ckpt_path = os.path.join(OUTPUT_DIR, "gpt2_giga_preprocess_progress.pt") # partial progress while preprocessing
save_path_100k = os.path.join(OUTPUT_DIR, "gpt2_giga_preprocessed.pt") # all labelled sentences
save_dir = OUTPUT_DIR                                                # checkpoints are saved here
TEXT_KEY = "document"     # dataset column with the article text
SUMMARY_KEY = "summary"   # dataset column with the reference summary


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

# Load GIGAWORD dataset
raw_dataset = load_dataset("gigaword")
train_articles = raw_dataset["train"]
full_val_split = raw_dataset["validation"]
val_data = full_val_split.select(range(2000))
test_data = full_val_split.select(range(2000, 12000))

# Preprocessing (skipped if the preprocessed file is already saved)
if os.path.exists(save_path_100k):
    print(f"Found preprocessed samples at {save_path_100k}")
    samples = torch.load(save_path_100k, map_location="cpu")
else:
    samples = []
    print(" Preprocessing 100k training samples with checkpointing...")
    for i in range(min(NUM_ARTICLES, len(train_articles))):
        article = train_articles[i]['document']
        summary = train_articles[i]['summary']
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

save_path = os.path.join(save_dir, "best_gpt2_extractive_gigaword.pt")
checkpoint_path = os.path.join(save_dir, "gpt2_extractive_checkpoint_gigaword.pt")

start_epoch = 0
best_rougel = 0.0
best_loss = float("inf")                    # training loss of the best epoch (breaks ties in ROUGE-L)
patience, no_improve, num_epochs = 3, 0, NUM_EPOCHS   # same early-stopping settings as the other scripts

if os.path.exists(checkpoint_path):
    print(" Resuming from checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_rougel = checkpoint['best_rougel']
    best_loss = checkpoint.get('best_loss', float("inf"))   # .get() so older checkpoints still load
    no_improve = checkpoint.get('no_improve', 0)
    start_epoch = checkpoint['epoch'] + 1


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

    val_rougel = compute_rougeL_on_validation(model, val_data, tokenizer, device, TEXT_KEY, SUMMARY_KEY, max_samples=NUM_VAL)
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


# Run test evaluation
evaluate_model(model, test_data, tokenizer, device, TEXT_KEY, SUMMARY_KEY, output_dir=OUTPUT_DIR, max_samples=NUM_TEST)
