"""Local Gradio demo for trained Thai toxicity classifiers."""
from pathlib import Path
import joblib
import gradio as gr
import torch
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.pipeline import thai_word_tokenize  # Registers the tokenizer function used by the saved baseline.

OUT = Path("outputs")
baseline = joblib.load(OUT / "baseline.joblib") if (OUT / "baseline.joblib").exists() else None
encoder_path = OUT / "wangchanberta"
tokenizer = AutoTokenizer.from_pretrained(encoder_path) if encoder_path.exists() else None
model = AutoModelForSequenceClassification.from_pretrained(encoder_path) if encoder_path.exists() else None
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
if model: model.to(device).eval()
thresholds_path = OUT / "thresholds.json"
thresholds = json.loads(thresholds_path.read_text()) if thresholds_path.exists() else {}

def selected_threshold(model_name):
    key = "baseline" if model_name.startswith("TF-IDF") else "wangchanberta"
    return float(thresholds.get(key, {}).get("threshold", .5))

def predict(text, model_name, threshold):
    if not text or not text.strip(): return {"Please enter text": 1.0}, ""
    if model_name.startswith("TF-IDF"):
        if baseline is None: return {}, "Run the pipeline first to train the baseline."
        p = float(baseline["classifier"].predict_proba(baseline["vectorizer"].transform([text]))[0,1])
    else:
        if model is None: return {}, "Run the pipeline without --skip-encoder to train WangchanBERTa."
        x = tokenizer(text, return_tensors="pt", truncation=True, max_length=192).to(device)
        with torch.no_grad(): p = float(torch.softmax(model(**x).logits, dim=-1)[0,1])
    t = float(threshold)
    return {"Toxic": p, "Non-toxic": 1-p}, f"Decision at threshold {t:.2f}: {'flag for human review' if p >= t else 'no flag'} (p(toxic)={p:.3f})."

with gr.Blocks(title="Thai Toxic Text Detection") as demo:
    gr.Markdown("# Thai toxic text detection\nEducational research demo. Scores are uncertain and must not be used for automatic moderation. Avoid entering identifying or sensitive text.")
    text=gr.Textbox(label="Thai text",lines=4,placeholder="Enter a short post")
    model_name=gr.Radio(["TF-IDF baseline","WangchanBERTa"],value="TF-IDF baseline",label="Model")
    threshold=gr.Slider(0,1,value=selected_threshold("TF-IDF baseline"),step=.01,label="Toxic decision threshold")
    button=gr.Button("Score")
    scores=gr.Label(label="Model probability"); decision=gr.Textbox(label="Interpretation",interactive=False)
    model_name.change(selected_threshold,inputs=model_name,outputs=threshold)
    button.click(predict,[text,model_name,threshold],[scores,decision])

if __name__ == "__main__": demo.launch()
