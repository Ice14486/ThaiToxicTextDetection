"""Reproducible Thai toxicity baseline and WangchanBERTa comparison."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

Path("work/cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path("work/matplotlib").resolve()))
os.environ.setdefault("PYTHAINLP_DATA", str(Path("work/pythainlp-data").resolve()))
os.environ.setdefault("HF_HOME", str(Path("work/cache/huggingface").resolve()))
os.environ.setdefault("HF_DATASETS_CACHE", str(Path("work/cache/huggingface/datasets").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("work/cache/xdg").resolve()))

import numpy as np
import pandas as pd
from datasets import load_dataset
from matplotlib import pyplot as plt
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.utils.class_weight import compute_class_weight
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)
import torch
from torch.utils.data import Dataset

SEED = 42
DATASET = "tmu-nlp/thai_toxicity_tweet"
DATASET_REVISION = "aa021e41d0ee6dbee2975fbed620ec8c586bdaf6"
ENCODER = "airesearch/wangchanberta-base-att-spm-uncased"
ENCODER_REVISION = "b81d38df6b4755dbedec0bfea863c9956cbb963e"
LABELS = ["non_toxic", "toxic"]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def fetch_data() -> pd.DataFrame:
    # The dataset repository uses a loading script; datasets 2.19.2 supports it.
    raw = load_dataset(DATASET, split="train", trust_remote_code=True, revision=DATASET_REVISION).to_pandas()
    raw = raw[raw.tweet_text.notna()].copy()
    raw = raw[raw.tweet_text.astype(str).str.strip().ne("TWEET_NOT_FOUND")]
    raw["text"] = raw.tweet_text.astype(str).str.strip()
    raw["label"] = raw.is_toxic.astype(int)
    raw["dedup_key"] = raw.text.str.replace(r"\s+", " ", regex=True).str.strip().str.casefold()
    raw = raw[raw.dedup_key.ne("")].drop_duplicates("dedup_key", keep="first")
    return raw.reset_index(drop=True)


def split_data(df: pd.DataFrame, out: Path) -> dict[str, np.ndarray]:
    # Persist once. Existing indices are reused to prevent accidental resplitting.
    p = out / "split_indices.json"
    if p.exists():
        obj = json.loads(p.read_text())
        if obj["n_rows"] != len(df):
            raise RuntimeError("Dataset row count changed; remove artifacts/split_indices.json deliberately to create a new split.")
        if obj.get("dataset_revision") not in (None, DATASET_REVISION):
            raise RuntimeError("Split artifact belongs to a different dataset revision.")
        if obj.get("dataset_revision") is None:
            obj["dataset_revision"] = DATASET_REVISION
            p.write_text(json.dumps(obj, indent=2))
        return {k: np.asarray(v, dtype=int) for k, v in obj["indices"].items()}
    ids = np.arange(len(df))
    train, rest = train_test_split(ids, test_size=.30, random_state=SEED, stratify=df.label)
    val, test = train_test_split(rest, test_size=.50, random_state=SEED, stratify=df.label.iloc[rest])
    parts = {"train": train, "validation": val, "test": test}
    p.write_text(json.dumps({"seed": SEED, "dataset_revision": DATASET_REVISION, "n_rows": len(df), "indices": {k: v.tolist() for k,v in parts.items()}}, indent=2))
    return parts


def describe_data(df: pd.DataFrame, parts: dict[str, np.ndarray], out: Path) -> None:
    stats = {"dataset": DATASET, "usable_rows_after_missing_text_and_dedup": len(df),
             "class_counts": {LABELS[int(k)]: int(v) for k,v in df.label.value_counts().sort_index().items()},
             "toxic_share": float(df.label.mean()),
             "text_char_length_median": float(df.text.str.len().median()),
             "text_char_length_p95": float(df.text.str.len().quantile(.95)),
             "splits": {k: {"n": len(v), "toxic_share": float(df.label.iloc[v].mean())} for k,v in parts.items()},
             "seed": SEED}
    (out / "data_summary.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    df.assign(char_length=df.text.str.len()).boxplot(column="char_length", by="label", ax=axes[0], grid=False)
    axes[0].set_title("Text length by label"); axes[0].set_xlabel("Label (0 non-toxic, 1 toxic)"); axes[0].set_ylabel("Characters")
    counts = df.label.value_counts().sort_index(); axes[1].bar(["Non-toxic", "Toxic"], [counts.get(0,0), counts.get(1,0)], color=["#6b8eae", "#d06b63"])
    axes[1].set_title("Class balance"); axes[1].set_ylabel("Posts")
    fig.suptitle(""); fig.tight_layout(); fig.savefig(out / "data_exploration.png", dpi=180); plt.close(fig)


def thai_word_tokenize(text):
    return word_tokenize(text, engine="newmm", keep_whitespace=False)


def baseline_features(train_text, other_text):
    word = TfidfVectorizer(tokenizer=thai_word_tokenize, token_pattern=None, lowercase=False,
                           ngram_range=(1,2), min_df=2, max_features=120_000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char", ngram_range=(2,5), min_df=2,
                           max_features=180_000, sublinear_tf=True)
    union = FeatureUnion([("word", word), ("char", char)])
    X_train = union.fit_transform(train_text)
    return union, X_train, union.transform(other_text)


def metric_row(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return {"macro_f1": float(f1_score(y, pred, average="macro")),
            "toxic_precision": float(precision_score(y, pred, zero_division=0)),
            "toxic_recall": float(recall_score(y, pred, zero_division=0)),
            "toxic_f1": float(f1_score(y, pred, zero_division=0)),
            "pr_auc": float(average_precision_score(y, prob)), "threshold": float(threshold),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0,1]).tolist()}


def best_f1_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    # Select the operating point using validation macro-F1, matching the headline metric.
    candidates = np.r_[0.0, thresholds, 1.0]
    scores = [f1_score(y, (p >= th).astype(int), average="macro") for th in candidates]
    idx = int(np.nanargmax(scores))
    return float(candidates[idx])


def plot_pr(y, predictions, out):
    plt.figure(figsize=(7,5))
    for name, p in predictions.items():
        pr, rc, _ = precision_recall_curve(y, p)
        plt.plot(rc, pr, label=f"{name} (AP={average_precision_score(y,p):.3f})")
    plt.xlabel("Toxic recall"); plt.ylabel("Toxic precision"); plt.title("Untouched test set: precision–recall")
    plt.legend(); plt.tight_layout(); plt.savefig(out / "precision_recall.png", dpi=180); plt.close()


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=192):
        self.enc = tokenizer(list(texts), truncation=True, padding="max_length", max_length=max_length)
        self.labels = list(map(int, labels))
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        d = {k: torch.tensor(v[i]) for k,v in self.enc.items()}; d["labels"] = torch.tensor(self.labels[i]); return d


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs); self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels"); outputs = model(**inputs)
        weights = self.class_weights.to(outputs.logits.device)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weights)
        return (loss, outputs) if return_outputs else loss


def fit_encoder(train_df, val_df, out: Path, epochs: int, batch: int):
    tokenizer = AutoTokenizer.from_pretrained(ENCODER, revision=ENCODER_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(ENCODER, revision=ENCODER_REVISION, num_labels=2, id2label={0:LABELS[0],1:LABELS[1]}, label2id={x:i for i,x in enumerate(LABELS)})
    train_ds=TextDataset(train_df.text, train_df.label, tokenizer); val_ds=TextDataset(val_df.text,val_df.label,tokenizer)
    cw=compute_class_weight("balanced", classes=np.array([0,1]), y=train_df.label.to_numpy())
    args=TrainingArguments(output_dir=str(out/"encoder_checkpoints"), num_train_epochs=epochs,
        per_device_train_batch_size=batch, per_device_eval_batch_size=batch*2,
        learning_rate=2e-5, weight_decay=.01, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss", save_total_limit=1,
        seed=SEED, data_seed=SEED, report_to="none", logging_steps=20,
        fp16=torch.cuda.is_available(), use_cpu=not (torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())))
    trainer=WeightedTrainer(model=model,args=args,train_dataset=train_ds,eval_dataset=val_ds,
        tokenizer=tokenizer,class_weights=torch.tensor(cw,dtype=torch.float))
    t0=time.perf_counter(); trainer.train(); train_seconds=time.perf_counter()-t0
    trainer.save_model(str(out/"wangchanberta")); tokenizer.save_pretrained(str(out/"wangchanberta"))
    return train_seconds


def perturb(text, kind):
    if not text: return text
    if kind == "repeat": return re.sub(r"(.)\1+", r"\1\1\1", text)
    if kind == "spacing": return re.sub(r"(?<=\S)(?=[ก-๙A-Za-z])", " ", text)
    if kind == "emoji": return text + " 😡"
    # Deterministic character deletion approximation of informal spelling variation.
    return re.sub(r"([ก-๙])\1", r"\1", text)


def error_group(text):
    t=text.lower()
    if re.search(r"[a-z]",t): return "mixed Thai-English"
    if re.search(r"(.)\1{2,}",t): return "repeated-character spelling"
    if re.search(r"555|ประชด|เหรอ|จ้า|🙄|😂",t): return "possible sarcasm / pragmatic cue"
    if re.search(r"\s",t): return "spacing / segmentation"
    return "slang, ambiguity, or missing context (manual review)"


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=Path("outputs")); ap.add_argument("--epochs",type=int,default=3); ap.add_argument("--batch-size",type=int,default=8); ap.add_argument("--skip-encoder",action="store_true"); args=ap.parse_args()
    seed_everything(); args.out.mkdir(parents=True,exist_ok=True)
    df=fetch_data(); parts=split_data(df,args.out)
    describe_data(df,parts,args.out)
    tr,va=(df.iloc[parts[k]].reset_index(drop=True) for k in ("train","validation"))
    # Train artifacts from train only. report_saved.py opens the test split after fitting is complete.
    t0=time.perf_counter(); vectorizer,Xtr,Xva=baseline_features(tr.text,va.text)
    clf=LogisticRegression(max_iter=1500,class_weight="balanced",solver="liblinear",random_state=SEED)
    clf.fit(Xtr,tr.label); baseline_train_s=time.perf_counter()-t0
    # Save baseline artifacts for the demo with an importable tokenizer reference.
    import joblib
    thai_word_tokenize.__module__ = "src.pipeline"
    joblib.dump({"vectorizer":vectorizer,"classifier":clf,"threshold":None},args.out/"baseline.joblib")
    training_times={"baseline":{"train_seconds":baseline_train_s}}
    if not args.skip_encoder:
        training_times["wangchanberta"]={"train_seconds":fit_encoder(tr,va,args.out,args.epochs,args.batch_size)}
    training_times.update({"seed":SEED,"dataset_revision":DATASET_REVISION,"encoder_revision":None if args.skip_encoder else ENCODER_REVISION,"epochs":0 if args.skip_encoder else args.epochs,"batch_size":args.batch_size})
    (args.out/"training_times.json").write_text(json.dumps(training_times,indent=2))
    print(json.dumps({"training_times":training_times,"data_summary":json.loads((args.out/"data_summary.json").read_text())},indent=2))

if __name__ == "__main__":
    thai_word_tokenize.__module__ = "src.pipeline"
    main()
