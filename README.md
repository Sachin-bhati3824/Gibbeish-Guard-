# GibberishGuard: A Character-Level Trigram Markov Model for Gibberish Detection

A statistical NLP project built **entirely from scratch** (Python + NumPy, no ML frameworks) that classifies text as fluent English or gibberish using a character-level trigram Markov language model — including manual implementations of Laplace smoothing, log-likelihood scoring, threshold optimization, and full evaluation metrics.

---

## Results at a Glance

![Score Distribution](./graphs/Score%20Distribution.png)

This is the most important plot in the project. It shows the average log-probability score distributions for three categories:

- **Correct** (real English) sentences cluster tightly near -3.5 to -4
- **Pure Gibberish** sentences spread broadly across -5 to -9
- **Hybrid** sentences (real words + gibberish words mixed) sit *in between*, heavily overlapping both distributions

This single graph visually explains the entire story of this project — see [The Hybrid Problem](#the-hybrid-problem-a-key-finding) below.

![ROC Curve](./graphs/Receiver%20Operating%20Characteristic%20(ROC)%20curve.png)
![Confusion Matrix](./graphs/Confusion%20Matrix.png)
![Bigram Transition Heatmap](./graphs/Bigram%20Transition%20Heatmap.png)
![Trigram Slice Heatmap (q)](./graphs/Trigram%20Slice%20Heatmap%20(Previous%20Character%20=%20'q').png)
![Trigram Frequency Distribution](./graphs/Trigram%20Frequency%20Distribution.png)

---

## The Journey

### Phase 1 — Bigram Model
Started with the simplest possible character-level model: `P(next_char | current_char)`. Built a 2D transition count matrix over a fixed character vocabulary, applied Laplace smoothing to remove zero probabilities, converted to log-probabilities, and scored text by average log transition probability. This gave a first working — but weak — gibberish detector.

### Phase 2 — Upgrading to Trigrams
Extended the model to `P(next_char | previous_char, current_char)` using a 3D NumPy count tensor (`counts[previous][current][next]`). This captured much richer character-pattern information and significantly improved the model's sense of "what English looks like."

### Phase 3 — Training at Scale
Trained on the [Kaggle English Sentences dataset](https://www.kaggle.com/datasets/mayakaripel/eng-sentences) — **~13 million English sentences**, ranging from 6-character fragments to full paragraphs, totaling roughly **100 million characters / ~100 MB**. Every sentence was normalized and its trigrams accumulated into the count tensor (`trigram_counts.npy`), then Laplace-smoothed and log-transformed into `trigram_log_probs.npy` for inference.

### Phase 4 — Scoring
Built `avg_transition_probability(text)`: walks every trigram in a string, looks up its log-probability, and averages across the sentence. Higher (less negative) average score = more English-like. Lower score = more gibberish-like.

### Phase 5 — Visualization
Visualized the trigram tensor with Matplotlib — 3D scatter plots, log-scaled frequency views (`np.log1p`), and 2D slices (e.g. "what follows `q`?") to sanity-check that the model had actually learned real linguistic structure (it had — `qu` dominates the `q`-slice heatmap, exactly as expected).

### Phase 6 — Building a Custom Benchmark
No existing dataset fit this task, so a 26,000-sample benchmark was built from scratch across three categories:

| Category | Count | Description |
|---|---|---|
| Correct | 13,000 | Real English sentences |
| Hybrid | 6,500 | Real words randomly mixed with gibberish words, shuffled |
| Pure Gibberish | 6,500 | Fully random character/word noise |

### Phase 7 — Threshold Optimization
Searched for the score cutoff that best separates English from gibberish:
```
score > threshold  → English
score <= threshold → Gibberish
```
Optimal threshold found experimentally: **-4.19**.

### Phase 8 — Full Evaluation Pipeline
Built confusion matrix, ROC curve, and every standard classification metric **manually** — no `sklearn.metrics`.

### Phase 9 — Baseline Exploration (Dictionary Ratio)
As a sanity check, also tried a much simpler approach: `valid_words / total_words` against a 50,000-word dictionary (`dictionlary_ratio.py`). This baseline scored *very well* — but it relies entirely on a predefined vocabulary and demonstrates nothing about statistical language modeling. **It was deliberately set aside** in favor of the trigram Markov model, which better demonstrates the core NLP/probability concepts this project was built to explore.

---

## The Hybrid Problem — A Key Finding

The single most interesting result of this project came from the hybrid category, which scored noticeably lower (71.6%) than the Correct (94.6%) and Pure Gibberish (95.4%) categories.

At first glance this looks like a model weakness. But digging into it revealed something more subtle:

> **I found that whole-sentence average log-likelihood conflates two different questions — "is this sentence gibberish *overall*?" vs "does this sentence *contain any* gibberish?" — and my hybrid benchmark is implicitly testing the second while my model answers the first.**

A hybrid sentence like `everyone sadjdfl and can fdsj` is mostly real English with a couple of injected nonsense words. A trigram model averaging log-probability across the *entire* sentence will correctly perceive it as "mostly fine," because that's what sentence-level average likelihood measures. The benchmark, however, labels it as gibberish because *some* tokens are fake.

So the 71.6% hybrid accuracy isn't purely "the model is bad at hybrids" — it's partly **the model and the benchmark disagreeing about what a hybrid sentence is**. Both framings are defensible, but this distinction matters: it points toward a concrete next step (per-word/windowed scoring instead of whole-sentence averaging) rather than just "the model needs to be better."

The Score Distribution graph above is the visual proof of this: the Hybrid distribution sits squarely in the overlap zone between Correct and Pure Gibberish — exactly where you'd expect a "mostly fine, a little gibberish" sentence to land under sentence-level averaging.

---

## Final Benchmarks

**Training corpus**: [Kaggle English Sentences](https://www.kaggle.com/datasets/mayakaripel/eng-sentences) — ~13 million sentences, ~100 million characters (`eng_sentences.tsv`)

**Evaluation benchmark**: 26,000 custom-built samples (13,000 Correct / 6,500 Hybrid / 6,500 Pure Gibberish)

### Per-Category Accuracy

| Category | Accuracy |
|---|---|
| Correct | 94.6308% |
| Hybrid | 71.6154% |
| Pure Gibberish | 95.4154% |

### Confusion Matrix

| | Predicted Positive (English) | Predicted Negative (Gibberish) |
|---|---|---|
| **Actual Positive (English)** | 12,302 | 698 |
| **Actual Negative (Gibberish)** | 2,143 | 10,857 |

### Overall Metrics

| Metric | Value |
|---|---|
| Accuracy | 89.073% |
| Precision | 85.164% |
| Recall | 94.631% |
| F1-Score | 89.648% |
| Specificity | 83.515% |
| ROC-AUC | 95.132% |
| Type 1 Error | 16.485% |
| Type 2 Error | 5.369% |

---

## Repository Contents

This repository contains everything needed to **reproduce the evaluation directly** — no retraining required.

- `trigram_counts.npy` / `trigram_log_probs.npy` — the trained model (raw counts and final smoothed log-probabilities)
- `vocabulary.pkl` — the words dataset built during the calculations of dictionary-ratio feature
- `check_the_acccuracy.py` — run this directly to reproduce all benchmarks, confusion matrix, and metrics on the included evaluation datasets
- `markove_chains_second_order_triagram.py` — trigram model training/scoring logic
- `markov_chains_first_order_bigram.py` — earlier bigram model (Phase 1)
- `dictionlary_ratio.py` — the alternative dictionary-ratio baseline (Phase 9)
- `correct_words.txt`, `gibbberish_words.txt`, `pure_gibberish.txt`, `hybrid_gibberish.txt`, `EnglishTenseUniqueDataset.tsv` — evaluation datasets used to build the 26K benchmark
- `pure_gibberish_gen.py`, `hybrid_gibberish_gen.py` — scripts used to generate the gibberish and hybrid evaluation sets
- `graphs/` — all visualizations shown above

**Note**: The training corpus (`eng_sentences.tsv`, ~100 MB, ~13M sentences) is **not included** in this repo due to size — download it from the [Kaggle link above](https://www.kaggle.com/datasets/mayakaripel/eng-sentences) only if you want to retrain from scratch. It is **not required** to reproduce the benchmarks; the trained `.npy` matrices are already included.

### Requirements
```
numpy
matplotlib
```
Standard library modules used: `pickle`, `mpl_toolkits.mplot3d`, `sys`, `random`

### Running the Benchmark
```bash
python check_the_acccuracy.py
```
This loads `trigram_log_probs.npy` and `vocabulary.pkl`, scores all samples in the evaluation datasets, and prints the confusion matrix and all metrics shown above.

---

## What This Project Demonstrates

- Character-level N-gram language models (bigram → trigram)
- Markov chains and probability distributions
- Laplace smoothing and log-probability scoring
- Threshold optimization for binary classification
- Manual implementation of confusion matrices, ROC curves, and all standard classification metrics
- Custom dataset construction and evaluation methodology
- Honest error analysis — including identifying and explaining a real limitation (the hybrid-sentence ambiguity above) rather than hiding it
