"""Evaluate saved artifacts on the fixed test split and create report outputs."""
import json
import gc
import os
import time
from pathlib import Path

Path("work/cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path("work/matplotlib").resolve()))
os.environ.setdefault("HF_HOME", str(Path("work/cache/huggingface").resolve()))
os.environ.setdefault("HF_DATASETS_CACHE", str(Path("work/cache/huggingface/datasets").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("work/cache/xdg").resolve()))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score)
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.pipeline import (DATASET, LABELS, SEED, baseline_features, best_f1_threshold,
    error_group, fetch_data, metric_row, perturb, plot_pr, split_data)

OUT = Path("outputs")
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def encoder_probs(texts, tokenizer, model, batch=16):
    ans=[]
    for i in range(0,len(texts),batch):
        enc=tokenizer(list(texts[i:i+batch]),return_tensors="pt",truncation=True,padding=True,max_length=192).to(DEVICE)
        with torch.no_grad(): ans.extend(torch.softmax(model(**enc).logits,dim=-1)[:,1].cpu().numpy().tolist())
    return np.asarray(ans)

def main():
    OUT.mkdir(exist_ok=True)
    df=fetch_data(); parts=split_data(df,OUT)
    summary={"dataset":DATASET,"usable_rows_after_missing_text_and_dedup":len(df),
        "class_counts":{LABELS[int(k)]:int(v) for k,v in df.label.value_counts().sort_index().items()},
        "toxic_share":float(df.label.mean()),"text_char_length_median":float(df.text.str.len().median()),
        "text_char_length_p95":float(df.text.str.len().quantile(.95)),
        "annotator_vote_split_2_to_1":int((df.toxic_votes.ne(df.nontoxic_votes)&df.toxic_votes.ne(3)&df.nontoxic_votes.ne(3)).sum()),
        "seed":SEED,"splits":{k:{"n":len(v),"toxic_share":float(df.label.iloc[v].mean())} for k,v in parts.items()}}
    (OUT/"data_summary.json").write_text(json.dumps(summary,indent=2))
    tr,va,te=(df.iloc[parts[k]].reset_index(drop=True) for k in ("train","validation","test"))
    training=json.loads((OUT/"training_times.json").read_text())
    t=time.perf_counter(); vec,Xtr,_=baseline_features(tr.text,va.text)
    clf=LogisticRegression(max_iter=1500,class_weight="balanced",solver="liblinear",random_state=SEED).fit(Xtr,tr.label)
    baseline_train=time.perf_counter()-t
    joblib.dump({"vectorizer":vec,"classifier":clf,"threshold":None},OUT/"baseline.joblib")
    t=time.perf_counter(); bval=clf.predict_proba(vec.transform(va.text))[:,1]; bval_time=time.perf_counter()-t
    bth=best_f1_threshold(va.label.to_numpy(),bval)
    t=time.perf_counter(); bprob=clf.predict_proba(vec.transform(te.text))[:,1]; btime=time.perf_counter()-t
    predictions={"baseline":bprob}; probs_by_label={"TF-IDF + Logistic Regression":bprob}
    raw={"baseline":metric_row(te.label.to_numpy(),bprob,bth)}
    raw["baseline"].update(train_seconds=baseline_train, test_inference_ms_per_item=btime*1000/len(te))
    thresholds={"baseline":{"validation_objective":"macro-F1","threshold":bth}}
    val_inference={"baseline":bval_time*1000/len(va)}
    has_encoder="wangchanberta" in training and (OUT/"wangchanberta").exists()
    if has_encoder:
        tkn=AutoTokenizer.from_pretrained(OUT/"wangchanberta"); model=AutoModelForSequenceClassification.from_pretrained(OUT/"wangchanberta").to(DEVICE).eval()
        t=time.perf_counter(); evalprob=encoder_probs(va.text.tolist(),tkn,model); evtime=time.perf_counter()-t
        eth=best_f1_threshold(va.label.to_numpy(),evalprob)
        t=time.perf_counter(); eprob=encoder_probs(te.text.tolist(),tkn,model); etime=time.perf_counter()-t
        predictions["wangchanberta"]=eprob; probs_by_label["WangchanBERTa"]=eprob
        raw["wangchanberta"]=metric_row(te.label.to_numpy(),eprob,eth)
        raw["wangchanberta"].update(train_seconds=training["wangchanberta"]["train_seconds"], test_inference_ms_per_item=etime*1000/len(te))
        thresholds["wangchanberta"]={"validation_objective":"macro-F1","threshold":eth}
        val_inference["wangchanberta"]=evtime*1000/len(va)
    (OUT/"results.json").write_text(json.dumps(raw,indent=2))
    (OUT/"thresholds.json").write_text(json.dumps(thresholds,indent=2))
    np.savez_compressed(OUT/"test_predictions.npz",y=te.label.to_numpy(),**predictions)
    if has_encoder: del model, tkn
    gc.collect()
    print("Saved test predictions and aggregate metrics.", flush=True)
    plot_pr(te.label.to_numpy(),probs_by_label,OUT)
    # Data exploration figure.
    fig,axes=plt.subplots(1,2,figsize=(10,4)); lengths=df.text.str.len()
    axes[0].boxplot([lengths[df.label.eq(0)],lengths[df.label.eq(1)]],tick_labels=["Non-toxic","Toxic"],showfliers=False)
    axes[0].set_title("Text length by label"); axes[0].set_ylabel("Characters")
    counts=df.label.value_counts().sort_index(); axes[1].bar(["Non-toxic","Toxic"],[counts.get(0,0),counts.get(1,0)],color=["#6b8eae","#d06b63"])
    axes[1].set_title("Class balance"); axes[1].set_ylabel("Posts")
    fig.tight_layout(); fig.savefig(OUT/"data_exploration.png",dpi=180); plt.close(fig)
    y=te.label.to_numpy(); texts=te.text.astype(str).tolist(); robustness={}; audit_summary={}
    for name,prob in predictions.items():
        threshold=thresholds[name]["threshold"]; pred=(prob>=threshold).astype(int)
        mask=pred != y; positions=np.flatnonzero(mask)
        err=te.iloc[positions].copy(); err["predicted_label"]=pred[positions]; err["confidence"]=np.maximum(prob[positions],1-prob[positions]); err["error_group_hint"]=err.text.map(error_group)
        err["toxic_votes"]=te.loc[positions,"toxic_votes"].to_numpy(); err["nontoxic_votes"]=te.loc[positions,"nontoxic_votes"].to_numpy()
        err["label_vote_margin"]=abs(err.toxic_votes-err.nontoxic_votes)
        err.head(50)[["text","label","predicted_label","confidence","error_group_hint","toxic_votes","nontoxic_votes","label_vote_margin"]].to_csv(OUT/f"{name}_errors_review.csv",index=False)
        # No test errors met the prespecified 0.90 confidence filter in this run.
        # Preserve a reviewable queue of the 50 most confident errors, with scores visible.
        audit=positions[np.argsort(-np.maximum(prob[positions],1-prob[positions]))[:50]]
        aud=te.iloc[audit].copy(); aud["predicted_label"]=pred[audit]; aud["confidence"]=np.maximum(prob[audit],1-prob[audit])
        aud["toxic_votes"]=te.iloc[audit].toxic_votes.to_numpy(); aud["nontoxic_votes"]=te.iloc[audit].nontoxic_votes.to_numpy(); aud["label_vote_margin"]=abs(aud.toxic_votes-aud.nontoxic_votes)
        aud["meets_90pct_confidence"]=aud.confidence.ge(.90)
        aud["human_label_review"]=""
        aud[["text","label","predicted_label","confidence","meets_90pct_confidence","toxic_votes","nontoxic_votes","label_vote_margin","human_label_review"]].to_csv(OUT/f"{name}_label_audit.csv",index=False)
        audit_summary[name]={"top_error_cases_queued":len(aud),"cases_at_least_90pct_confident":int(aud.meets_90pct_confidence.sum()),
            "cases_with_2_to_1_annotation_vote":int(aud.label_vote_margin.eq(1).sum()),"median_model_confidence":float(aud.confidence.median()) if len(aud) else None,
            "human_review_completed":False}
        robustness[name]={}
    if has_encoder:
        model=AutoModelForSequenceClassification.from_pretrained(OUT/"wangchanberta").to(DEVICE).eval()
        tkn=AutoTokenizer.from_pretrained(OUT/"wangchanberta")
    for name,prob in predictions.items():
        threshold=thresholds[name]["threshold"]
        for kind in ("repeat","spacing","emoji","spelling"):
            changed=[perturb(x,kind) for x in texts]
            if name=="baseline": changed_prob=clf.predict_proba(vec.transform(changed))[:,1]
            else: changed_prob=encoder_probs(changed,tkn,model)
            robustness[name][kind]=float(recall_score(y[y==1],(changed_prob[y==1]>=threshold).astype(int),zero_division=0))
    (OUT/"robustness.json").write_text(json.dumps(robustness,indent=2))
    (OUT/"audit_summary.json").write_text(json.dumps(audit_summary,indent=2))
    fig,ax=plt.subplots(figsize=(8,4.5)); cats=["original","repeat","spacing","emoji","spelling"]
    for name,vals in robustness.items():
        start=raw[name]["toxic_recall"]; ax.plot(cats,[start,*vals.values()],marker="o",label=name)
    ax.set_ylabel("Toxic recall"); ax.set_ylim(0,1); ax.set_title("Recall under deterministic text perturbations"); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"robustness.png",dpi=180); plt.close(fig)
    print(json.dumps({"results":raw,"thresholds":thresholds,"robustness":robustness,"audit_summary":audit_summary,"validation_inference_ms_per_item":val_inference},indent=2))

if __name__=="__main__": main()
