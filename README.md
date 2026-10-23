# Thai Toxic Text Detection: baseline vs. fine-tuned encoder

> **Key finding:** On the fixed 324-post test set, the TF-IDF baseline outperformed a one-epoch WangchanBERTa run (macro-F1 **0.732 vs. 0.687**; toxic recall **0.770 vs. 0.690**). For this run, the simpler model also trained in **0.7 s** versus **345 s** on CPU. This is an initial comparison; the encoder received only one epoch on this CPU-only host.

This project compares a Thai token/character TF-IDF logistic-regression baseline with class-weighted WangchanBERTa. It evaluates validation-selected decision thresholds on a held-out test set and measures sensitivity to simple text perturbations. The model is a research aid; predictions need human review.

## Data and split

The source is [`tmu-nlp/thai_toxicity_tweet`](https://huggingface.co/datasets/tmu-nlp/thai_toxicity_tweet). Its card describes 3,300 tweets originally annotated by three annotators, with 2,027 toxic and 1,273 non-toxic labels. Of the 3,300, 506 have text replaced by `TWEET_NOT_FOUND`; the pipeline excludes those and removes exact normalized-text duplicates. This run retained **2,160 unique posts**: **1,332 toxic (61.7%)** and **828 non-toxic**. Character length was median **75** and 95th percentile **139**. Among usable posts, **344 (15.9%)** had a 2–1 annotator vote split, a direct disagreement signal but not proof that the majority label is wrong. Counts and split shares are in `outputs/data_summary.json`; `outputs/data_exploration.png` visualizes length and class balance. The source card reports a CC BY-NC 3.0 license. Use the dataset and any derived data only under its terms; do not redistribute the raw text.

The pipeline creates one stratified 70/15/15 train/validation/test split using seed 42 and persists row indices in `outputs/split_indices.json`. It pins the dataset revision (`aa021e4`) and WangchanBERTa revision (`b81d38d`), reuses the split on later runs, and stops if the usable row count changes. Model training and threshold selection use train and validation only. The separate report command opens the test split after training. Generated review queues contain raw posts and are git-ignored; keep them local and handle them carefully.

## Models and setup

- **Baseline:** PyThaiNLP `newmm` word tokens with word 1–2 grams, combined with character 2–5 grams; balanced logistic regression.
- **Main model:** `airesearch/wangchanberta-base-att-spm-uncased`, fine-tuned with class-weighted cross-entropy.
- **Threshold:** selected on validation by maximizing toxic-class F1. The threshold is then frozen for test and robustness metrics. For moderation, lower thresholds catch more toxic posts but generate more false flags; teams should choose an operating point from validation precision/recall and send flags to human review.
- **Timing:** reports training seconds and test inference milliseconds per item on the machine used. These timings are hardware dependent.

Use Python 3.10–3.13 (the pinned `datasets` loader relies on a dataset loading script):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline --epochs 1 --batch-size 16
python -m src.report_saved
python app.py
```

The first command trains the models; the report command evaluates the saved checkpoints on the fixed test split and writes the figures and review queues. Run `python -m src.pipeline --epochs 3 --batch-size 8` for a longer encoder fine-tune. Run `python -m src.pipeline --skip-encoder` for only the baseline. Training downloads the dataset and pretrained weights from Hugging Face. A GPU is recommended; the one-epoch CPU fit took 345 seconds here. The optional external LLM comparison is omitted because it requires a separately configured API, and no cost/latency claims are made without one.

## Results

`outputs/results.json` records macro-F1, toxic precision/recall/F1, PR-AUC (average precision), confusion matrix, validation-selected threshold, training time, and end-to-end inference speed. `outputs/precision_recall.png` plots test precision–recall curves. Aggregate JSON and shareable figures are also copied to `reports/` for version control. These results use the single test split and thresholds selected on validation by macro-F1.

| Model | Macro-F1 | Toxic precision | Toxic recall | Toxic F1 | PR-AUC | Threshold | Train sec | Test ms/item |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + logistic regression | 0.732 | 0.806 | 0.770 | 0.788 | 0.883 | 0.527 | 0.7 | 0.154 |
| WangchanBERTa (1 epoch) | 0.687 | 0.789 | 0.690 | 0.736 | 0.851 | 0.545 | 344.9 | 9.519 |

The confusion matrices use `[ [TN, FP], [FN, TP] ]`: baseline `[[87, 37], [46, 154]]`; encoder `[[87, 37], [62, 138]]`. On this run, the validation-selected baseline threshold is the better measured operating point: it catches 154 of 200 toxic test posts with 37 false flags among 124 non-toxic posts. A lower threshold would prioritize catching more toxic posts and increase false flags; set the production point from validation requirements and route flags to human review.

## Analysis findings

`outputs/robustness.json` and `outputs/robustness.png` report toxic recall after deterministic repeated-character, spacing, emoji, and spelling perturbations. Baseline recall was 0.770 on the original test posts, 0.780 after repeated-character changes, 0.535 after spacing changes, 0.780 after an angry-face emoji suffix, and 0.770 after collapsing repeated letters. Encoder recall was 0.690, 0.695, 0.975, 0.810, and 0.695 in the same order. The spacing operation inserts spaces between adjacent Thai/Latin characters and is intentionally severe; these are synthetic stress tests, not realistic prevalence estimates.

The pipeline exports 50 misclassified examples per model to local `*_errors_review.csv` files with rough review hints (mixed Thai-English, repeated spelling, possible sarcasm, or ambiguity/context). The hints are automated and unvalidated. The label-audit queue contains the 50 highest-confidence errors per model and includes all three source annotator votes. **No test error reached 90% model confidence in this run**; the 50 queued items are ranked review candidates, with median confidence about 0.61. Among those ranked cases, 10 baseline and 9 encoder examples had 2–1 source annotation votes. These vote splits indicate annotator disagreement, not a human estimate of mislabeled examples. The queues have not been manually adjudicated, so no mislabeled-case rate is claimed. Do not paste offensive examples into public documentation; describe categories generally or mask words.

## Limitations and responsible use

The source labels reflect three annotators and majority vote; the dataset card notes disagreement around sarcasm, unclear targets, and ambiguous word senses. Toxicity judgments depend on context and culture. The tweets date from 2017 and the unavailable-text exclusion may introduce selection bias. Perturbations are synthetic and limited. A classifier can miss harm or falsely flag benign text; it is not suitable for automated moderation without human review. The dataset license is non-commercial (CC BY-NC 3.0).

## How to reproduce

1. Install dependencies and run `python -m src.pipeline` from the repository root.
2. Keep `outputs/split_indices.json` to reuse the original split. To deliberately create a new split, remove that file and document the change.
3. The generated outputs are ignored by Git so raw text, weights, and local audit examples are not accidentally committed. Share aggregate metrics and figures only after reviewing them for privacy and accuracy.
4. Launch the local demo with `python app.py` after training.
