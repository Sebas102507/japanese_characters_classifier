# Japanese KMNIST Character Classifier — Full Pipeline Analysis

## Overview

This project trains fully connected neural networks (MLPs) in PyTorch to classify handwritten Japanese Hiragana characters from the **Kuzushiji-MNIST (KMNIST)** dataset. The dataset contains **70,000 grayscale images** of 10 different Hiragana characters, each of dimension 28×28 pixels, yielding a flattened input vector of 784 features. The 10 classes are distributed nearly uniformly, with each class representing approximately 10% of the data.

The assignment constraint restricts models to **fully connected and dropout layers only** — no convolutional layers were used. The goal was to run at least 3 experiments and achieve a model with **≥80% validation accuracy** with limited overfitting.

Eight experiments were conducted in total: one dummy baseline and seven neural network configurations.

---

## 1. Data Loading

The dataset was sourced from Kaggle ([kuzushiji dataset](https://www.kaggle.com/datasets/anokas/kuzushiji/data)) as four `.npz` files:

| File | Description |
|---|---|
| `kmnist-train-imgs.npz` | 60,000 training images |
| `kmnist-train-labels.npz` | 60,000 training labels |
| `kmnist-test-imgs.npz` | 10,000 test images |
| `kmnist-test-labels.npz` | 10,000 test labels |

Each image is a 28×28 grayscale matrix. Labels are integers from 0 to 9, each corresponding to one of 10 Japanese Hiragana characters.

---

## 2. Data Preprocessing

### 2.1 Tensor Conversion

Raw NumPy arrays were converted to PyTorch float32 tensors and pushed to the appropriate device (CUDA if available, otherwise CPU).

### 2.2 Data Augmentation

To increase training set diversity and improve generalisation, **three augmented copies** of the original training set were generated and concatenated:

| Augmentation | Description |
|---|---|
| Rotation | Random rotation up to ±15 degrees |
| Affine | Random rotation + translation up to 15% of image size |
| Scaling | Random rotation + scale factor between 0.8× and 1.2× |

This quadrupled the effective training set size (original + 3 augmented copies).

### 2.3 Flattening and Normalisation

Each 28×28 image was flattened to a vector of 784 values. Pixel intensities were normalised from their original range of [0, 255] to [0, 1] by dividing by 255.

### 2.4 One-Hot Encoding

Target labels were converted to a one-hot binary class matrix of shape `(N, 10)`. For example:
- Class 0 → `[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]`
- Class 5 → `[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]`

### 2.5 Train / Validation / Test Split

The original 10,000-sample test set was split 50/50 into a validation set and a held-out test set:

| Split | Size |
|---|---|
| Training | 240,000 (60,000 × 4 after augmentation) |
| Validation | 5,000 |
| Test | 5,000 |

Class distributions across all three splits were verified visually via bar charts and confirmed to be balanced.

---

## 3. Model Architecture

All experiments used the same base architecture class `JapaneseCharactersNN`, a configurable fully connected MLP with:

- A variable number of hidden layers with configurable widths
- **ReLU** activations after each hidden layer
- **Dropout** applied after each hidden layer (configurable probability)
- A final linear output layer of size 10 (one per class)

The PyTorch seed was fixed to 42 for reproducibility.

---

## 4. Training Infrastructure

A custom `Experiment` class managed the full training loop with:

- **PyTorch Ignite** engines for training and evaluation steps
- **Early stopping** with configurable patience (default: 20 epochs) based on validation loss
- **Best model checkpointing**: the model state at the epoch with the lowest validation loss was saved and restored at the end of training
- **Optional LR scheduler**: `ReduceLROnPlateau` was used in later experiments to automatically decay the learning rate when validation loss stagnated
- Per-epoch tracking of train/validation loss and accuracy

A custom `ExperimentPlotter` class provided:
- Loss and accuracy history curves
- Confusion matrices
- Misclassification diagnostic panels (failed samples vs. correctly classified examples of both classes)

---

## 5. Experiments

### Experiment 0 — Dummy Baseline

**Strategy:** Most-frequent class dummy classifier (always predicts the most common class regardless of input).

| Metric | Result |
|---|---|
| Train Accuracy | 10.00% |
| Validation Accuracy | 10.32% |

**Analysis:**

The dummy (most-frequent) classifier achieves **10.00% train accuracy** and **10.32% validation accuracy**, which reflects a near perfectly balanced 10-class dataset — each class makes up roughly 10% of the data. Because the strategy always predicts the single most frequent class, it is correct only when the true label happens to be that class and gets everything else wrong.

**Key takeaways:**

- **Random-chance ceiling.** On a balanced 10-class problem, always guessing the majority class gives exactly 1/10 = 10% accuracy. The dummy model hits this exactly, confirming the KMNIST training split is nearly uniform across all 10 Japanese characters.

- **Zero discriminative power.** The dummy model ignores all pixel information — it carries no feature learning whatsoever.

- **Establishes the lower bound.** Any model that cannot beat ~10% accuracy on this task has failed to learn anything meaningful. All subsequent experiments (Experiments 1–6) must comfortably exceed this threshold to be considered useful, with the assignment target set at 80%.

---

### Experiment 1 — Shallow Baseline Network

**Architecture:** `[64, 32, 16]` — 3 hidden layers  
**Optimizer:** Adam, lr=0.01  
**Regularisation:** Dropout p=0.5  
**Criterion:** CrossEntropyLoss  
**Early Stopping:** patience=20, best epoch: 8 (stopped at epoch 28)

| Metric | Result |
|---|---|
| Train Accuracy | ~66% |
| Validation Accuracy | ~54% |
| Train/Val Gap | ~12 pp |

**Analysis:**

The model reached a best validation loss at **epoch 8** (early stopping triggered at epoch 28), with roughly **~66% train accuracy** and **~54% validation accuracy** at the restored checkpoint.

**vs. Dummy baseline (10%):**

Experiment 1 is a substantial step above the dummy classifier — validation accuracy improves from 10% to ~54%, confirming the network has learnt real discriminative features from the pixel data. However, the gap between train (~66%) and validation (~54%) indicates moderate **overfitting**.

**Loss & accuracy curves:**

- Training loss steadily decreases while validation loss remains high and noisy — a classic overfitting signature.
- Validation accuracy is highly volatile across epochs, suggesting the model is sensitive to its initialisation and the relatively aggressive learning rate (lr=0.01).
- Early stopping correctly prevents further overfitting by restoring the best checkpoint from epoch 8.

**Confusion matrix observations:**

- Classes **0, 3, 4, 7, 9** are relatively well recovered on the training set, showing the model has learnt some structure for simpler characters.
- Class **2** is the weakest — a large portion of its samples are misclassified as class **8**, suggesting visual similarity between those two characters that the shallow network cannot resolve.
- Class **1** is frequently confused with classes **4** and **6**, again pointing to visually similar stroke patterns.
- On the **validation** set the same confusion patterns appear but are more pronounced, reinforcing the overfitting diagnosis.

---

### Experiment 2 — Deeper Architecture

**Architecture:** `[512, 256, 128, 64, 32, 16]` — 6 hidden layers  
**Optimizer:** Adam, lr=0.001  
**Regularisation:** Dropout p=0.5 (no weight decay)  
**Criterion:** CrossEntropyLoss  
**Early Stopping:** patience=20, best epoch: 39 (stopped at epoch 59)

| Metric | Result |
|---|---|
| Train Accuracy | ~93% |
| Validation Accuracy | ~83% |
| Train/Val Gap | ~10 pp |

**Analysis:**

The deeper [512, 256, 128, 64, 32, 16] network reached its best validation loss at **epoch 39** (early stopping at epoch 59), achieving roughly **~93% train accuracy** and **~83% val accuracy**.

**vs. Dummy baseline (10%) and Experiment 1 (~54% val):**

This is the first experiment to surpass the 80% target. Validation accuracy jumps from ~54% (Experiment 1) to ~83% — a gain of nearly **30 percentage points** driven entirely by the increased network capacity. The much deeper architecture can capture more complex non-linear boundaries between visually similar Japanese characters. The dummy baseline (10%) is now left far behind.

**Loss & accuracy curves:**

- Both train and val loss decrease sharply in the first ~10 epochs before plateauing, indicating fast initial learning.
- Training loss continues to fall steadily while validation loss flattens and becomes noisy after ~epoch 10 — the hallmark of **overfitting** starting to set in.
- The train/val accuracy gap (~10 pp: 93% vs 83%) is still notable, suggesting the model is memorising some training patterns. No explicit regularisation (e.g. weight decay) is applied here, which explains this gap.
- Early stopping correctly picks epoch 39, well before validation loss starts climbing back up.

**Confusion matrix observations:**

- The diagonal is dense and dominant across all 10 classes on both train and val sets — a major improvement over Experiment 1.
- The most persistent confusion on validation is **class 2 → class 6** (21 mispredicted as 6) and **class 5 → class 2** (63 mispredicted as 2), characters with structurally similar strokes.
- **Class 7** shows some leakage into class 9 (36 errors) and class 4 (32 errors).
- **Class 0** still has notable confusion with class 4 (85 errors), mirroring the issue seen in Experiment 1.

---

### Experiment 3 — Deep Architecture with L2 Regularisation

**Architecture:** `[512, 256, 128, 64, 32, 16]` — same as Experiment 2  
**Optimizer:** Adam, lr=0.001, weight_decay=0.001  
**Regularisation:** Dropout p=0.5 + L2 weight decay  
**Criterion:** CrossEntropyLoss  
**Early Stopping:** patience=20, best epoch: 137 (stopped at epoch 157)

| Metric | Result |
|---|---|
| Train Accuracy | ~89% |
| Validation Accuracy | ~78% |
| Train/Val Gap | ~11–12 pp |

**Analysis:**

Same [512, 256, 128, 64, 32, 16] architecture as Experiment 2 but with L2 weight decay added. Best validation loss reached at **epoch 137** (early stopping at epoch 157), with roughly **~89% train accuracy** and **~78% val accuracy**.

**vs. Dummy baseline (10%), Experiment 1 (~54% val), Experiment 2 (~83% val):**

Counterintuitively, Experiment 3 performs **worse on validation than Experiment 2** (~78% vs ~83%), despite adding regularisation. This suggests the weight decay penalty was too aggressive for this architecture, constraining the model's capacity and preventing it from converging to the same quality solution as Experiment 2.

**Loss & accuracy curves:**

- Both loss curves decrease much more slowly and over a longer horizon (157 epochs vs 59 in Experiment 2) — a direct effect of L2 penalising large weights and slowing optimisation.
- The train/val accuracy gap is **larger here (~11–12 pp: ~89% vs ~78%)** than in Experiment 2, which is the opposite of what regularisation aims to achieve.
- Validation accuracy is noisier across epochs compared to Experiment 2, oscillating with high variance even in the later stages.
- Early stopping at epoch 137 is appropriate but the model never converged to a stable plateau.

**Confusion matrix observations:**

- The diagonal is still dominant but noticeably lighter than in Experiment 2 — more predictions are scattered off-diagonal.
- **Class 2** is the weakest link, with heavy leakage into class 2 for true classes 6 (69), 7 (83), and 9 (75) on the validation set.
- **Class 3 → class 8** confusion (57 errors on val) and **class 7 → class 4** (64 errors) are more pronounced than in Experiment 2.
- **Class 0 → class 7** confusion (62 errors on val) is a new pattern not prominent in Experiment 2, suggesting the overpenalised model is losing discriminative boundaries.

---

### Experiment 4 — Wide Shallow + LR Scheduler (Best Model)

**Architecture:** `[512, 256]` — 2 hidden layers  
**Optimizer:** Adam, lr=0.001, weight_decay=0.00015  
**Regularisation:** Dropout p=0.4 + light L2 + ReduceLROnPlateau (factor=0.1, patience=10)  
**Criterion:** CrossEntropyLoss  
**Early Stopping:** patience=20, best epoch: 53 (stopped at epoch 73)

| Metric | Result |
|---|---|
| Train Accuracy | ~99.6% |
| Validation Accuracy | ~93% |
| Train/Val Gap | ~7 pp |

**Analysis:**

A simplified [512, 256] architecture with lighter L2 regularisation and a learning rate scheduler. Best validation loss reached at **epoch 53** (early stopping at epoch 73), achieving roughly **~99.6% train accuracy** and **~93% val accuracy**.

**vs. previous experiments:**

This is the best result so far — a **+10 pp jump over Experiment 2** and well clear of the 80% target. The key drivers are: (1) a wider but shallower architecture that concentrates capacity in fewer, larger layers, and (2) a learning rate scheduler allowing the optimiser to settle into a sharper minimum.

**Loss & accuracy curves:**

- Train loss drops very steeply in the first ~10 epochs and then continues declining slowly down to near zero by epoch 73. The scheduler's LR decay is visible as the additional step down in train loss around epoch 50.
- Val loss decreases at the beginning, then diverges and plateaus around 0.25 — the train/val gap has widened significantly (~7 pp: 99.6% vs 93%), indicating the model is strongly memorising the training set.
- Val accuracy stabilises cleanly around 93% with much less noise than previous experiments, showing the scheduler helps smooth convergence.
- The large train accuracy (~99.6%) relative to val is a clear **overfitting** signal.

**Confusion matrix observations:**

- The training confusion matrix is nearly a perfect diagonal — virtually no off-diagonal errors across all 10 classes, consistent with near-perfect train accuracy.
- The **validation** matrix is also notably clean, with most classes achieving 450–490 correct out of ~500 samples.
- Remaining errors on validation are small and well distributed: the largest confusion pairs are **class 2 → class 3** (27) and **class 1 → class 6** (24).
- No single class dominates the errors, suggesting the model has learnt a balanced representation across all 10 Japanese characters.

**This model was selected for final evaluation on the held-out test set.**

---

### Experiment 5 — Wider Regularisation with Label Smoothing

**Architecture:** `[512, 256]` — same as Experiment 4  
**Optimizer:** Adam, lr=0.001, weight_decay=0.0005  
**Regularisation:** Dropout p=0.4 + stronger L2 + label smoothing=0.1  
**Criterion:** CrossEntropyLoss (label_smoothing=0.1)  
**Early Stopping:** patience=20, best epoch: 88 (stopped at epoch 108)

| Metric | Result |
|---|---|
| Train Accuracy | ~98.5% |
| Validation Accuracy | ~91.4% |
| Train/Val Gap | ~7 pp |

**Analysis:**

Same [512, 256] architecture and scheduler as Experiment 4, but with stronger regularisation via label smoothing. Best validation loss at **epoch 88** (early stopping at epoch 108), reaching roughly **~98.5% train accuracy** and **~91.4% val accuracy**.

**vs. Experiment 4 (~99.6% train / ~93% val):**

Experiment 5 trades a small amount of validation accuracy (~93% → ~91.4%) for a meaningfully **reduced train/val gap** (~7 pp in Exp 4 vs ~7 pp here). The gap does not narrow as much as expected, suggesting label smoothing alone is not sufficient to close the generalisation deficit. Notably, train accuracy drops from 99.6% to ~98.5%, showing label smoothing is successfully preventing the model from reaching near-zero training loss — but the regularisation effect does not fully transfer to the validation set.

**Loss & accuracy curves:**

- The two clear step-downs in both loss curves (around epochs 40 and 88) are the LR scheduler firing — each step causes a sharp improvement on both train and val metrics.
- Val loss drops from ~0.85 to ~0.73 after the second scheduler step, showing the lower LR helps the model into a better minimum.
- Val accuracy stabilises around 91–92% after epoch 40 with low noise, and then steps up to ~91.4% after epoch 88.
- The train/val gap remains visible but the curves are more parallel than in Experiment 4, suggesting slightly better generalisation behaviour.

**Confusion matrix observations:**

- The training confusion matrix is still very strongly diagonal but with slightly more visible noise than Experiment 4, consistent with label smoothing preventing overconfident predictions.
- On **validation**, the matrix is slightly weaker than Experiment 4: class **1 → class 6** confusion (32 errors) and **class 2 → class 3** (27 errors) persist from Experiment 4.
- **Class 5** shows increased errors into class 2 (15) and class 6 (12) compared to Experiment 4 — a regression in that class.
- **Class 7** deteriorates compared to Experiment 4, with more leakage into class 9 (10) and class 4 (9).
- Overall, error counts per class are slightly higher than Experiment 4 across the board.

---

### Experiment 6 — Heavier Regularisation

**Architecture:** `[512, 256]` — same as Experiments 4 and 5  
**Optimizer:** Adam, lr=0.001, weight_decay=0.0005  
**Regularisation:** Dropout p=0.5 + L2 + label smoothing=0.15  
**Criterion:** CrossEntropyLoss (label_smoothing=0.15)  
**Early Stopping:** patience=20, best epoch: 89 (stopped at epoch 109)

| Metric | Result |
|---|---|
| Train Accuracy | ~97.8% |
| Validation Accuracy | ~90.7% |
| Train/Val Gap | ~7 pp |

**Analysis:**

Same [512, 256] architecture and scheduler as Experiments 4 and 5, but with increased label smoothing (0.15) and higher dropout (p=0.5). Best validation loss at **epoch 89** (early stopping at epoch 109), reaching roughly **~97.8% train accuracy** and **~90.7% val accuracy**.

Experiment 6 continues the trend of stronger regularisation reducing train accuracy, but the train/val gap is not meaningfully narrowed. Val accuracy drops slightly further (~90.7%) compared to Experiments 4 and 5. The combined effect of higher label smoothing and dropout is pushing training harder, but the validation benefit has not materialised.

**Loss & accuracy curves:**

- The two LR scheduler step-downs are again clearly visible (~epoch 25 and ~epoch 88), producing sharp improvements in both train and val metrics each time.
- Train and val loss curves are more closely spaced than in Experiments 4/5 — the higher regularisation is compressing the gap in loss space, even if the accuracy gap remains similar.
- Val accuracy stabilises at ~90–91% with low noise after epoch 40, then steps up marginally after epoch 88.
- Train accuracy is noticeably lower than in Experiment 4 (~97.8% vs ~99.6%), confirming the stronger dropout is effectively preventing memorisation.

**Confusion matrix observations:**

- The training matrix shows more off-diagonal scatter than in Experiments 4 and 5, which is expected given the higher dropout — the model is less certain on training samples.
- On **validation**, the matrix remains clean overall, but with slightly lower diagonal values than Experiment 4.
- **Class 1 → class 6** confusion (33 errors) persists as the most notable error, unchanged from Experiment 5 — indicating this is a fundamental visual ambiguity in the data, not a regularisation issue.
- **Class 2 → class 3** (28 errors) also remains stable across experiments.
- **Class 7** continues to struggle slightly (leakage into class 4 with 12 errors, class 9 with 10 errors).
- Most classes achieve 430–476 correct out of ~500 validation samples, with class 7 (432) and class 9 (431) being the weakest.

Experiment 6 confirms that pushing regularisation further (higher label smoothing + dropout) does not improve generalisation beyond Experiment 4's ~93% — it only trades training accuracy for a marginal and inconsistent change in validation performance.

---

### Experiment 7 — Compact Network with Scheduler

**Architecture:** `[64, 32]` — 2 hidden layers, much smaller  
**Optimizer:** Adam, lr=0.001, weight_decay=0.00015  
**Regularisation:** Dropout p=0.4 + light L2 + ReduceLROnPlateau (factor=0.1, patience=10)  
**Criterion:** CrossEntropyLoss  
**Early Stopping:** patience=20, best epoch: 58 (stopped at epoch 78)

| Metric | Result |
|---|---|
| Train Accuracy | ~91% |
| Validation Accuracy | ~80.7% |
| Train/Val Gap | ~10 pp |

**Analysis:**

The compact [64, 32] network with ReduceLROnPlateau scheduler, light L2, and dropout p=0.4 reached its best validation loss at **epoch 58** (early stopping at epoch 78), achieving roughly **~91% train accuracy** and **~80.7% val accuracy**.

Experiment 7 just clears the 80% target. Compared to Experiment 4 (same scheduler strategy), the drastically smaller architecture ([64, 32] vs [512, 256]) costs ~12 pp of validation accuracy.

**Loss & accuracy curves:**

- Both loss curves decrease steadily throughout training with no sharp scheduler-triggered steps — ReduceLROnPlateau fires reactively on stagnation rather than at fixed intervals, producing a smoother, more gradual decline.
- The train/val loss gap widens progressively after epoch ~10 — a clear overfitting signal consistent with the smaller network not having enough inductive capacity to generalise.
- Val accuracy climbs from ~73% to ~80.7% over 78 epochs, stabilising with moderate noise in the final third — the scheduler is doing useful work but the model has hit a capacity ceiling.
- Train accuracy reaches ~91%, notably **lower than in Experiments 4–6**, reflecting that even on training data the small network cannot perfectly fit the patterns — it is under-parameterised rather than over-parameterised.

**Confusion matrix observations:**

- The training confusion matrix shows significantly more off-diagonal scatter than Experiments 4–6, consistent with the model's limited capacity.
- **Class 4** is the weakest on training (994 errors), with heavy leakage into class 0 — a pattern not seen in wider networks, suggesting the two bottleneck layers cannot fully separate these characters.
- **Class 2** on training shows substantial confusion with class 6 (1466), class 8 (882), and class 3 (499).
- On **validation**, the most notable errors are **class 1 → class 6** (47), **class 2 → class 3** (29), and **class 7 → class 4** (35) — the same persistent ambiguous pairs seen across all experiments.
- **Class 7** (341/500) and **class 2** (370/500) are the weakest on validation, both significantly below the ~450–490 correct seen in Experiment 4.

Experiment 7 confirms that a compact [64, 32] architecture is insufficient to match the wider models — it barely meets the 80% threshold and introduces more per-class confusion. The ReduceLROnPlateau scheduler is a sound strategy (smooth, reactive LR decay) but cannot compensate for the capacity bottleneck. **Experiment 4 remains the best model overall**.

---

## 6. Results Summary

| Experiment | Architecture | LR | Weight Decay | Label Smooth | Dropout | Best Epoch | Train Acc | Val Acc |
|---|---|---|---|---|---|---|---|---|
| 0 (Dummy) | — | — | — | — | — | — | 10.0% | 10.32% |
| 1 | [64, 32, 16] | 0.01 | None | None | 0.5 | 8 | ~66% | ~54% |
| 2 | [512, 256, 128, 64, 32, 16] | 0.001 | None | None | 0.5 | 39 | ~93% | ~83% |
| 3 | [512, 256, 128, 64, 32, 16] | 0.001 | 0.001 | None | 0.5 | 137 | ~89% | ~78% |
| **4** ★ | **[512, 256]** | **0.001** | **0.00015** | **None** | **0.4** | **53** | **~99.6%** | **~93%** |
| 5 | [512, 256] | 0.001 | 0.0005 | 0.10 | 0.4 | 88 | ~98.5% | ~91.4% |
| 6 | [512, 256] | 0.001 | 0.0005 | 0.15 | 0.5 | 89 | ~97.8% | ~90.7% |
| 7 | [64, 32] | 0.001 | 0.00015 | None | 0.4 | 58 | ~91% | ~80.7% |

★ Selected model for final evaluation.

---

## 7. Misclassification Analysis (Selected Model: Experiment 4)

### Class 1 → Class 6 (24 validation errors)

The diagnostic plot reveals that **24 validation samples of class 1** were misclassified as **class 6**. Comparing the three panels:

- **Failed (Real 1 → Pred 6):** The misclassified characters tend to be written in a more compact, cursive style with prominent curved and crossing strokes in their lower half — the same structural region that defines class 6. Some examples show unusual stroke proportions or heavy ink density that obscure the characteristic horizontal arm of class 1.
- **Correct Real 6:** These correctly identified class 6 characters share the very same curved lower loop that the model latched onto in the misclassified class 1 samples, confirming the feature overlap that drives the confusion.
- **Correct Real 1:** When classified correctly, class 1 characters display a more clearly defined crossing horizontal stroke at the top and a separated lower element — features distinct enough from class 6 for the model to separate them.

The 1→6 confusion is a visual ambiguity in this dataset. Both characters share a rounded, multi-stroke lower region, and the model (a flat MLP operating on raw pixel values) cannot exploit spatial structure the way a CNN would. With only 24 val failures out of the full class 1 set, this is a minor but structurally motivated error mode rather than a sign of undertrained capacity.

### Class 2 → Class 3 (27 validation errors — most frequent error pair)

The diagnostic plot reveals that **27 validation samples of class 2** were misclassified as **class 3**. Comparing the three panels:

- **Failed (Real 2 → Pred 3):** The misclassified class 2 characters are written in a tightly looped, highly cursive style where the defining strokes collapse into a compact rounded form. The dominant visual signal in these samples is a single sweeping curve or closed loop. Heavy ink density and overlapping strokes further suppress the secondary descending element that normally separates class 2 from class 3.
- **Correct Real 3:** These correctly identified class 3 characters consistently display a clean, open or semi-open curved stroke with a smooth flowing tail, giving the model a reliable template to match against. The structural simplicity of class 3 means even slight visual overlap with a poorly formed class 2 sample is enough to tip the prediction.
- **Correct Real 2:** When class 2 is classified correctly, the characters retain a more complex multi-stroke arrangement — a vertical or angled primary stroke combined with a clearly separated secondary element — features distinct enough from the smooth single-curve profile of class 3 for the model to differentiate them.

With 27 validation failures, this is the most frequent single pair error in the model and reflects a fundamental limit of a flat MLP working on raw flattened pixels — it cannot reason about stroke order or local spatial arrangement the way a CNN or sequence model could, making it susceptible whenever two classes share a dominant low-frequency shape.

---

## 8. Final Test Set Evaluation

The selected model (Experiment 4) was evaluated on the held-out test set of 5,000 samples (kept completely separate from training and validation throughout all experiments).

| Split | Accuracy |
|---|---|
| Training | ~99.6% |
| Validation | ~93% |
| **Test** | **~93%** |

The test accuracy closely matches validation accuracy, confirming that the model generalises consistently and was not inadvertently tuned to the validation set. The confusion matrix on the test set mirrors the validation set patterns, with the same persistent but minor class confusions (1→6, 2→3).

---

## 9. Conclusions

### Key Findings

**Architecture capacity is the dominant driver of performance.** The single largest improvement across all experiments came from increasing network capacity in Experiment 2 — a jump from ~54% to ~83% validation accuracy purely through a deeper, wider architecture with no other changes.

**Width outperforms depth for this task.** Experiment 4's 2-layer `[512, 256]` network achieved better results (~93%) than the 6-layer `[512, 256, 128, 64, 32, 16]` network of Experiments 2 and 3 (~83% and ~78%). Concentrating representational capacity in fewer, larger transformations is more effective than distributing it across many narrow layers.

**Adaptive learning rate scheduling is highly effective.** Introducing `ReduceLROnPlateau` (Experiment 4 onwards) was a key differentiator. It enabled the optimiser to automatically find sharper minima when validation loss stagnated, producing cleaner and more stable convergence than fixed learning rates.

**Regularisation must be carefully calibrated.** Too little regularisation (Experiments 1, 2) leads to overfitting. Too much regularisation (Experiment 3 with weight_decay=0.001; Experiments 5 and 6 with strong label smoothing + dropout) constrains model capacity and reduces validation performance. The sweet spot — light L2 (weight_decay=0.00015) and moderate dropout (p=0.4) — was found in Experiment 4.

**Excessive regularisation yields diminishing returns.** Experiments 5 and 6 showed that pushing regularisation beyond Experiment 4's configuration only trades train accuracy for no meaningful improvement in generalisation. The dominant error patterns (1→6, 2→3 class confusions) stem from visual ambiguity in the dataset, not from overfitting — stronger regularisation cannot fix them.

**Compact architectures have hard capacity ceilings.** Experiment 7's `[64, 32]` network, despite using the same scheduler strategy as Experiment 4, barely cleared the 80% target and trailed by ~12 pp. Training strategy alone cannot compensate for insufficient architectural capacity.

**Data augmentation provided a useful regularisation signal.** Quadrupling the training set via geometric augmentations (rotation, affine, scale) gave the model richer variation to learn from, likely contributing to the stable generalisation seen from Experiment 4 onwards.

### Persistent Limitations

- All models are **fully connected MLPs operating on flattened 28×28 inputs**, discarding the 2D spatial structure of the characters. Convolutional architectures, which preserve local spatial relationships and stroke neighbourhoods, were not explored and would likely improve performance significantly.
- The most persistent confusion pairs (Class 1→6, Class 2→3) reflect **genuine visual ambiguity** in how different writers render certain Hiragana characters, a problem that spatial reasoning (CNNs, attention) would handle better than flat pixel vectors.
- No systematic robustness analysis was performed (noise, adversarial perturbations, rotation extremes beyond the augmentation range).

### Recommendations

1. **Evaluate convolutional neural networks** as the natural next step — CNNs retain local spatial structure and are inherently better suited to character image recognition.
2. **Systematically examine misclassified samples** by class pair to understand whether errors are caused by label noise, ambiguous handwriting styles, or genuine structural overlap between characters.
3. **Investigate mislabelled samples** in the training and validation sets; label noise can distort both training dynamics and the interpretation of per-class error patterns.
4. **Tune early stopping patience** — some experiments (e.g., Experiment 3) may have benefited from a longer patience window given the slower convergence under stronger regularisation.
