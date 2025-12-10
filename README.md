# Transformer-Based Extractive Summarization Study

A Comprehensive Evaluation of BERT, RoBERTa, BART, FLAN-T5, and GPT-2 on CNN/DailyMail and Gigaword

## Overview

This repository contains the full codebase, training scripts, evaluation notebooks, and qualitative comparison tools for a large-scale study of **extractive text summarization** using multiple transformer encoder architectures.

The project evaluates five major pretrained models:

- **BERT-base-uncased**
- **RoBERTa-base**
- **BART-base (encoder)**
- **FLAN-T5-small (encoder)**
- **GPT-2 (decoder-only)**

Each model is trained using a unified pipeline for sentence-level extractive summarization and evaluated on **CNN/DailyMail** and **Gigaword** datasets.

This study includes:

- End-to-end training pipelines
- Automatic evaluation (ROUGE, METEOR, BERTScore)
- Qualitative comparison notebooks
- A full research report summarizing findings

---

## Repository Structure

'''

transformer-extractive-summarization-study/
│
├── BART_CNN.py
├── BART_Eval_CNN.ipynb
├── BART_Eval_GIGA.ipynb
├── BART_GIGA.py
│
├── BERT_BART_ROBERTA_FLAN_T5_GPT2_QUALITATIVE_CNNDAILYMAIL.ipynb
├── BERT_BART_ROBERTA_FLAN_T5_GPT2_QUALITATIVE_GIGAWORD.ipynb
│
├── BERT_CNN.py
├── Bert_Eval_CNN.ipynb
├── Bert_Eval_GIGA.ipynb
├── BERT_GIGA.py
│
├── comprehensive_extractive_summarization_transformers_report.pdf
│
├── FLAN_T5_CNN.py
├── FLAN_T5_SMALL_CNN_EVAL.ipynb
├── FLAN_T5_SMALL_GIGA_Eval.ipynb
├── FLAN_T5_GIGA.py
│
├── GPT-2_CNN.ipynb
├── GPT2_Giga.py
├── GPT2-giga.ipynb
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── Roberta_CNN.py
├── Roberta_Eval_CNN.ipynb
├── Roberta_Eval_GIGA.ipynb
└── Roberta_Giga.py

'''

---

## Installation

Clone the repo:

```bash
git clone https://github.com/<your-username>/transformer-extractive-summarization-study.git
cd transformer-extractive-summarization-study

```

Install dependencies:

pip install -r requirements.txt

Download NLTK data (if needed):

import nltk
nltk.download("punkt")
nltk.download("wordnet")

## Datasets

This project uses HuggingFace Datasets:
• CNN/DailyMail

from datasets import load_dataset
load_dataset("cnn_dailymail", "3.0.0")

    •	Gigaword

load_dataset("gigaword")

All preprocessing happens dynamically inside the training scripts.

## Notes

    •	Paths are intentionally left blank for privacy; users must update them in each script.
    •	All models use sentence-level classification with ROUGE-L-based labeling.
    •	The repository represents an independent personal research project, fully runnable on local hardware or cloud environments.

## Training

Run training scripts directly for each model / dataset pair.

### CNN/DailyMail

The following scripts train sentence-level extractive models on CNN/DailyMail:

```bash
python BERT_CNN.py
python Roberta_CNN.py
python BART_CNN.py
python FLAN_T5_CNN.py
```

For GPT-2 on CNN/DailyMail, training and evaluation are combined in a single notebook:
• GPT-2_CNN.ipynb – runs preprocessing, training, final test evaluation, and saves 100 qualitative examples.

You can open it with:
jupyter notebook GPT-2_CNN.ipynb

Gigaword:
For Gigaword, most models are trained via standalone scripts:
python BERT_GIGA.py
python Roberta_Giga.py
python BART_GIGA.py
python FLAN_T5_GIGA.py

For GPT-2 on Gigaword, training and evaluation are combined in the notebook:
• GPT-2giga.ipynb (or GPT2-giga.ipynb depending on your filename) – runs preprocessing, training, validation, final test evaluation, and qualitative sample export.

Launch with:
jupyter notebook GPT-2giga.ipynb

---

### 🔁 Updated **Evaluation** section

Replace your current **`## Evaluation`** section with this:

````markdown
## Evaluation

Evaluation is done either via dedicated notebooks or inside the training notebooks, depending on the model.

For most models, evaluation notebooks compute:

- **ROUGE-1 / ROUGE-2 / ROUGE-L**
- **METEOR**
- **BERTScore (P / R / F1)**

Examples:

```bash
jupyter notebook BERT_Eval_CNN.ipynb
jupyter notebook Bert_Eval_GIGA.ipynb
jupyter notebook BART_Eval_CNN.ipynb
jupyter notebook BART_Eval_GIGA.ipynb
jupyter notebook Roberta_Eval_CNN.ipynb
jupyter notebook Roberta_Eval_GIGA.ipynb
jupyter notebook FLAN_T5_SMALL_CNN_EVAL.ipynb
jupyter notebook FLAN_T5_SMALL_GIGA_Eval.ipynb
```
````

For GPT-2, there are no separate evaluation notebooks:
• GPT-2_CNN.ipynb (CNN/DailyMail) and
• GPT-2giga.ipynb (Gigaword)

both perform training and final evaluation in the same notebook, including metric computation and qualitative summary export.

## Results Summary

| Model   | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | METEOR | BERTScore F1 |
| ------- | ---------- | ---------- | ---------- | ------ | ------------ |
| BERT    | 0.44       | 0.21       | 0.42       | 0.28   | 0.87         |
| RoBERTa | 0.45       | 0.22       | 0.43       | 0.29   | 0.88         |
| BART    | 0.46       | 0.22       | 0.44       | 0.29   | 0.88         |
| FLAN-T5 | 0.42       | 0.20       | 0.40       | 0.27   | 0.86         |
| GPT-2   | 0.40       | 0.18       | 0.38       | 0.26   | 0.85         |

| Model   | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | METEOR | BERTScore F1 |
| ------- | ---------- | ---------- | ---------- | ------ | ------------ |
| BERT    | 0.44       | 0.21       | 0.42       | 0.28   | 0.87         |
| RoBERTa | 0.45       | 0.22       | 0.43       | 0.29   | 0.88         |
| BART    | 0.46       | 0.22       | 0.44       | 0.29   | 0.88         |
| FLAN-T5 | 0.42       | 0.20       | 0.40       | 0.27   | 0.86         |
| GPT-2   | 0.40       | 0.18       | 0.38       | 0.26   | 0.85         |

## Full Research Report

The full PDF report is available:

comprehensive_extractive_summarization_transformers_report.pdf

It includes:
• Literature review
• Model architecture overview
• Training methodology
• Quantitative results
• Error analysis
• Conclusions

## License

This repository is released under the MIT License.

## Acknowledgements

Special thanks to:
• HuggingFace for transformers and datasets
• NLTK for linguistic tools
• ROUGE & BERTScore authors
• CNN/DailyMail and Gigaword dataset creators
