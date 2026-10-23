# Thai Toxic Text Detection: baseline vs. fine-tuned encoder

> **Key finding:** On the fixed 324-post test set, TF-IDF reached macro-F1 **0.732** and toxic recall **0.770**. WangchanBERTa trained for **3 epochs** reached macro-F1 **0.804** and toxic recall **0.880** using the **epoch-2 checkpoint**, which had the lowest validation loss (0.4443; epoch 3 was 0.4475). The baseline trained in about **0.8 seconds**; the full encoder run took about **374 seconds on Apple MPS**. This is an initial comparison, not a production moderation system.

The complete workflow is in [thai_toxicity_analysis.ipynb](thai_toxicity_analysis.ipynb). It covers dataset cleaning, the fixed split, class/text exploration, both models, validation threshold selection, held-out test evaluation, error/label-audit queues, robustness checks, plots, and an optional Gradio demo. Run its cells from top to bottom. The test split is accessed only after training and validation threshold selection.

## Data and split

The original 3,300-row export is stored locally at data/thai_toxicity_raw.csv for easy browsing. The notebook loads this raw CSV and visibly cleans it: it removes missing text, unavailable-text placeholders, blank posts, and normalized duplicate posts, showing the row count after every step. This reproduces the 2,160 usable posts used in the experiment. The raw CSV is excluded from Git because it contains posts; use it under the dataset license and do not redistribute it.

Source: [tmu-nlp/thai_toxicity_tweet](https://huggingface.co/datasets/tmu-nlp/thai_toxicity_tweet). The card describes 3,300 tweets, 2,027 toxic and 1,273 non-toxic, originally annotated by three annotators. It reports CC BY-NC 3.0. Of the 3,300, 506 have unavailable text; after removing those and normalized duplicates, this run retained **2,160 posts**: **1,332 toxic (61.7%)** and **828 non-toxic**. Median length is **75** characters and the 95th percentile is **139**. The stratified 70/15/15 split uses seed 42 and is saved in outputs/split_indices.json.

## Results

| Model | Macro-F1 | Toxic precision | Toxic recall | Toxic F1 | PR-AUC | Threshold | Train seconds | Test ms/item |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + logistic regression | 0.732 | 0.806 | 0.770 | 0.788 | 0.883 | 0.527 | 0.8 | 0.233 |
| WangchanBERTa (3 epochs; epoch-2 checkpoint selected) | 0.804 | 0.834 | 0.880 | 0.856 | 0.931 | 0.454 | 374.1 | 3.629 |

Thresholds are selected on validation by macro-F1 and then frozen for the test set. The encoder checkpoint is selected by validation loss; the epoch-2 checkpoint was restored after the 3-epoch run. Confusion matrices use [[TN, FP], [FN, TP]]: baseline [[87, 37], [46, 154]], encoder [[89, 35], [24, 176]]. Lower thresholds tend to catch more toxic posts and create more false flags; moderation decisions should route candidates to human review.

## Analysis and limitations

The notebook measures recall after synthetic repeated-character, spacing, emoji, and spelling perturbations. It creates local review queues with up to 50 errors per model and the most confident error cases. The cause labels are heuristic hints, not human-reviewed findings. No manual label adjudication was completed, so no mislabeled-case rate is claimed. Do not publish raw-text review CSVs.

Labels are subjective and depend on culture and context; annotators disagreed on a portion of the data. The tweets date from 2017, unavailable-text exclusion may introduce selection bias, and perturbations are synthetic. The classifier is not suitable for automated moderation without human review. Optional LLM comparison was not run because it requires a configured API.

## How to run

1. Open the notebook and select the project virtual environment as its kernel.
2. If needed, install dependencies with `pip install -r requirements.txt`.
3. Run notebook cells from top to bottom. The encoder cell defaults to 3 epochs and keeps the checkpoint with the lowest validation loss. Runtime depends on hardware; the saved run used Apple MPS and took about 374 seconds. Model and dataset caches are stored under outputs/cache/.
4. To use the Gradio demo, run the last cell and uncomment its `demo.launch` line.

The notebook displays the report inline and saves machine-readable results, figures, split indices, review queues, and trained weights under outputs/. Raw review queues and model weights are git-ignored. The notebook uses the local raw CSV when available. If it is absent, the pinned dataset revision is downloaded from Hugging Face; the pretrained checkpoint is also downloaded if it is not cached. Dataset and derived data use must comply with CC BY-NC 3.0; do not redistribute raw text.
