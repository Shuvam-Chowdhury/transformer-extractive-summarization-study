#!/usr/bin/env python
# coding: utf-8
"""Shared evaluation code used by every training script.

Every model is validated and tested the same way:
- split the article into sentences (pretrained NLTK Punkt, same as training)
- score each sentence with the model
- keep the 3 highest-scoring sentences (or all of them if the article has fewer than 3)
- put those sentences back in the order they appear in the article
"""

import os
import csv
import json

import torch
from nltk.tokenize import sent_tokenize
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score

TOP_K = 3          # number of sentences in each predicted summary
MAX_LENGTH = 128   # tokens per sentence, same as training


def predict_summary(model, tokenizer, article, device, top_k=TOP_K):
    """Pick the top_k sentences of one article and return them in article order."""
    sentences = sent_tokenize(article)                  # same splitter as training
    if len(sentences) == 0:
        return ""                                       # empty article, nothing to pick

    tokenized = tokenizer(sentences, padding="max_length", truncation=True,
                          max_length=MAX_LENGTH, return_tensors="pt")
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    logits = model(input_ids, attention_mask).reshape(-1)  # one score per sentence, even for a 1-sentence article
    scores = logits.tolist()

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)  # best score first
    top_indices = ranked[:top_k]                         # keeps all sentences if there are fewer than top_k
    top_indices = sorted(top_indices)                    # put them back in article order

    chosen = []
    for i in top_indices:
        chosen.append(sentences[i])
    return " ".join(chosen)


def compute_rougeL_on_validation(model, dataset, tokenizer, device, text_key, summary_key, max_samples=2000):
    """Average ROUGE-L F1 on the validation articles (used to pick the best epoch)."""
    model.eval()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    total_rougeL = 0.0
    n = 0
    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            article = dataset[i][text_key]
            reference = dataset[i][summary_key]
            prediction = predict_summary(model, tokenizer, article, device)
            if prediction == "":
                continue                                 # skip only empty articles
            total_rougeL += scorer.score(reference, prediction)["rougeL"].fmeasure
            n += 1
    if n == 0:
        return 0.0
    return total_rougeL / n                              # average over the articles actually scored


def evaluate_model(model, dataset, tokenizer, device, text_key, summary_key, output_dir=None, max_samples=10000):
    """Final test: ROUGE-1/2/L F1, METEOR and BERTScore P/R/F1.

    Prints the metrics, returns them as a dict, and (if output_dir is given) saves
    test_metrics.json and test_predictions.csv in output_dir.
    """
    model.eval()
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1_total = 0.0
    rouge2_total = 0.0
    rougeL_total = 0.0
    meteor_total = 0.0
    references = []
    predictions = []
    row_indices = []
    skipped_empty = 0

    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            article = dataset[i][text_key]
            reference = dataset[i][summary_key]
            prediction = predict_summary(model, tokenizer, article, device)
            if prediction == "":
                skipped_empty += 1                       # skip only empty articles
                continue

            scores = rouge.score(reference, prediction)
            rouge1_total += scores["rouge1"].fmeasure
            rouge2_total += scores["rouge2"].fmeasure
            rougeL_total += scores["rougeL"].fmeasure
            meteor_total += single_meteor_score(reference.split(), prediction.split())  # METEOR needs word lists

            references.append(reference)
            predictions.append(prediction)
            row_indices.append(i)

    n = len(predictions)
    if n == 0:
        print("\nNo articles could be scored.", flush=True)
        return None

    precision, recall, f1 = bert_score(predictions, references, lang="en", verbose=False)

    metrics = {
        "articles_scored": n,
        "articles_skipped_empty": skipped_empty,
        "rouge1_f1": rouge1_total / n,
        "rouge2_f1": rouge2_total / n,
        "rougeL_f1": rougeL_total / n,
        "meteor": meteor_total / n,
        "bertscore_precision": precision.mean().item(),
        "bertscore_recall": recall.mean().item(),
        "bertscore_f1": f1.mean().item(),
    }

    print(f"\n Final Evaluation on {n} samples ({skipped_empty} empty articles skipped)", flush=True)
    print(f"ROUGE-1 F1: {metrics['rouge1_f1']:.4f}", flush=True)
    print(f"ROUGE-2 F1: {metrics['rouge2_f1']:.4f}", flush=True)
    print(f"ROUGE-L F1: {metrics['rougeL_f1']:.4f}", flush=True)
    print(f"METEOR:     {metrics['meteor']:.4f}", flush=True)
    print(f"BERTScore P/R/F1: {metrics['bertscore_precision']:.4f} / "
          f"{metrics['bertscore_recall']:.4f} / {metrics['bertscore_f1']:.4f}", flush=True)

    if output_dir is not None:
        metrics_path = os.path.join(output_dir, "test_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        predictions_path = os.path.join(output_dir, "test_predictions.csv")
        with open(predictions_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["test_index", "reference", "prediction"])
            for j in range(n):
                writer.writerow([row_indices[j], references[j], predictions[j]])

        print(f"Saved {metrics_path} and {predictions_path}", flush=True)

    return metrics
