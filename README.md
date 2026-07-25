# Skin Lesion Analysis — ISIC 2018

![Python](https://img.shields.io/badge/Python-3.7%E2%80%933.9-3776AB?logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.5-FF6F00?logo=tensorflow&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-green.svg)

Deep learning on dermoscopy images for the [ISIC 2018 "Skin Lesion Analysis Towards Melanoma Detection"](https://challenge.isic-archive.com/landing/2018/) challenge, covering all three official tasks: lesion **boundary segmentation**, **attribute detection**, and **disease classification**.

> **This was my first deep learning project** (2021–2022). I've kept it as it was built — including the parts that didn't work — and added an honest write-up of what I'd do differently now. See [What I'd do differently](#what-id-do-differently).

## Results

Measured numbers, taken from the notebooks' own saved outputs:

| ISIC task | Notebook | Model | Metric | Score |
|---|---|---|---|---|
| **Task 1** — lesion segmentation | [`task1`](task1-lesion-segmentation.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/teddy8023mars/Skin-lesion-classification/blob/main/task1-lesion-segmentation.ipynb) | ResNet50-U-Net | Jaccard / Dice / Accuracy | **0.793** / 0.873 / 0.933 |
| **Task 2** — attribute detection | [`task2`](task2-attribute-detection.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/teddy8023mars/Skin-lesion-classification/blob/main/task2-attribute-detection.ipynb) | ResNet50 multi-label U-Net | pooled Jaccard (macro) | **0.200** |
| **Bonus** — concept bottleneck | [`cbm`](concept-bottleneck-pipeline.ipynb) | Task2 attributes → interpretable classifier | *see notebook* | *implementation, not yet trained* |
| **Task 3** — lesion classification | [`task3`](task3-lesion-classification.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/teddy8023mars/Skin-lesion-classification/blob/main/task3-lesion-classification.ipynb) | EfficientNetB0 @224 (2026)<br><sub>2021: small CNN @32 — leaked</sub> | Balanced acc / macro-F1 / AUC | **0.703** / 0.655 / 0.944 |

<sub>Task 1 also reports recall 0.906 / precision 0.871. Task 2 is the unweighted mean of five per-attribute pooled Jaccards — see [the metric trap](#a-metric-trap-in-the-task-2-literature) before comparing it to other papers. Task 3's headline accuracy is 0.801, but the majority-class baseline is 0.669, so balanced accuracy is the honest figure; **melanoma sensitivity is only 0.419** and that is the number that matters. The 2021 notebook's widely-quoted 61.5% is invalid — it leaked. Full metrics: [`assets/task2-results.json`](assets/task2-results.json), [`assets/task3-results.json`](assets/task3-results.json).</sub>

---

## Task 1 — Lesion boundary segmentation

Pixel-wise segmentation of the lesion region: the first step of any automated dermoscopy pipeline.

**Encoder ablation.** Rather than accepting one architecture, I trained and compared six:

![Segmentation results across six architectures](assets/segmentation-results.png)

U-Net (from scratch), VGG16-U-Net, VGG19-U-Net, DenseNet121-U-Net, Double U-Net, and ResNet50-U-Net — all on the same image, against ground truth. Plain U-Net and ResNet50-U-Net produce a spurious second blob here; the ImageNet-pretrained VGG/DenseNet encoders are cleaner. ResNet50-U-Net gave the best overall validation scores and became the primary model.

**Hair-removal preprocessing.** Dermoscopy images are full of occluding body hair, which U-Net happily segments as lesion edge. A black-hat morphology + inpainting pass removes it before training (`u_net_with_hair_removal`).

**Other techniques:** combined Dice + binary cross-entropy loss, Jaccard-optimized variant, test-time augmentation (horizontal / vertical / both flips), Dice / IoU / recall / precision tracked per epoch.

## Task 2 — Lesion attribute detection

> **Added in 2026.** Task 2 was skipped when I first built this project — which is why the notebooks used to jump from 1 to 3. It is now implemented **and trained**, on the official 2594-image Task 2 training set (split 1816 / 389 / 389, seed 42), on a free Kaggle P100.

Detect five dermoscopic attributes as **per-pixel binary masks**, using the same input images as Task 1 (hence the `t12` in the dataset paths):

| Attribute | Mask file suffix | Present in training set |
|---|---|---|
| Pigment network | `_attribute_pigment_network.png` | 58.7% of images |
| Globules | `_attribute_globules.png` | 24.4% |
| Milia-like cysts | `_attribute_milia_like_cyst.png` | 26.2% |
| Negative network | `_attribute_negative_network.png` | 7.5% |
| Streaks | `_attribute_streaks.png` | 3.9% |

<sub>Frequencies from Le et al., *TATL: Task Agnostic Transfer Learning for Skin Attributes Detection*, Medical Image Analysis 2022 ([arXiv:2104.01641](https://arxiv.org/abs/2104.01641)).</sub>

**Why this is much harder than Task 1.** A lesion boundary is one large, high-contrast, always-present object. These attributes are faint, tiny, scattered, **co-occurring**, and often absent — streaks appear in under 4% of images. Design consequences:

- **Five independent sigmoids, not softmax.** Attributes co-occur in the same pixel, so this is multi-label segmentation: output shape `(256, 256, 5)`.
- **Loss must fight sparsity.** Combined Dice + binary cross-entropy, with positive weights *measured from the data* in the EDA cell rather than hard-coded.
- **Nearest-neighbour mask resizing.** Bilinear interpolation puts grey fringes on these structures and erases the smallest cysts outright.
- **Pooled Jaccard, not per-image average.** The official metric pools every prediction pixel across the whole dataset, precisely because many images have no positive pixels for a given attribute — per-image averaging would be undefined or misleading. The notebook implements the official pooled version.
- **Per-attribute threshold search.** Heavy positive weighting makes 0.5 the wrong operating point; thresholds are fitted on validation only.

### Measured result

Trained on a free Kaggle P100, ResNet50-U-Net encoder, 30 epochs, ~30 minutes. Scored on the held-out 389-image test split with per-attribute thresholds fitted on validation only:

![Per-attribute Jaccard vs the 2018 winner](assets/task2-jaccard-vs-winner.png)

| Attribute | Threshold | **This U-Net** | 2018 winner | Present in train | Positive pixels |
|---|---|---|---|---|---|
| Pigment network | 0.90 | **0.410** | 0.563 | 58.1% | 4.08% |
| Globules | 0.85 | **0.289** | 0.341 | 22.4% | 0.59% |
| Negative network | 0.95 | **0.131** | 0.228 | 7.3% | 0.25% |
| Streaks | 0.95 | **0.107** | 0.156 | 4.1% | 0.067% |
| Milia-like cysts | 0.90 | **0.064** | 0.171 | 26.8% | 0.215% |
| **Macro average** | | **0.200** | **0.292** | | |

Three things worth extracting from this:

**1. Threshold tuning was worth +37%.** Macro Jaccard at the default 0.5 was 0.146; with per-attribute thresholds fitted on validation it is 0.200. Every fitted threshold landed in 0.85–0.95, far from 0.5 — a direct consequence of positive weights up to 50× pushing the sigmoid up. A model evaluated at 0.5 here would look much worse than it is.

**2. The difficulty ordering matches the winner's exactly** (pigment network best, milia-like cysts and streaks worst), which says this ranking is a property of the task rather than of the model. Pigment network occupies 4.08% of pixels; streaks 0.067% — a **61× difference in pixel mass**.

**3. "Rare" has two meanings, and the pixel-level one is what hurts.** Milia-like cysts appear in 26.8% of images — far more often than negative network's 7.3% — yet score worst of all five. They are common but *minuscule*: scattered specks totalling 0.215% of pixels. Image-level frequency and pixel-level sparsity are different problems.

**Under-trained, not converged.** Validation dice was still improving at epoch 30 and early stopping never fired, so 0.200 is what 30 epochs bought, not this architecture's ceiling.

![Training curves](assets/task2-training-curves.png)

### What the model actually learned — and why that matters downstream

![Attribute predictions vs ground truth](assets/task2-attribute-results.png)

Ground truth uses all five colours; the predictions are dominated by one coarse blue (globules) contour tracing roughly the whole lesion, with the fine pigment-network mesh largely missed. (These four are the most densely annotated test images, deliberately picked, so they are not typical — but the direction is real.)

The honest reading: **the model has likely learned a coarse "where is the lesion" prior more than genuine attribute-texture discrimination.** Because attribute masks mostly sit inside the lesion, predicting "the whole lesion" already earns a non-trivial Jaccard. This is a concrete, testable hypothesis, and it is exactly what the concept-fidelity check in the [concept bottleneck pipeline](concept-bottleneck-pipeline.ipynb) exists to catch: concepts that encode lesion area rather than dermoscopic structure would make an "interpretable" bottleneck interpretable in name only.

### A metric trap in the Task 2 literature

Published Task 2 numbers are not comparable without knowing how they aggregate across attributes, and papers frequently do not say.

- The **0.292** above is the winner's *unweighted mean of the five per-attribute Jaccards* (Le et al., [arXiv:2104.01641](https://arxiv.org/abs/2104.01641)). That is the same aggregation used for this repo's 0.200, so those two are comparable.
- The official ISIC guidance only specifies pooling **across images** — "computing the Jaccard over all pixels in the set, rather than image-by-image" ([ISIC forum](https://forum.isic-archive.com/t/task-2-evaluation-and-superpixel-generation/417)) — and does not state how the five attributes are combined.
- Other published Task 2 figures sit on a visibly different scale: a multi-task U-Net reports **0.433 for 5th place** ([chvlyl/ISIC2018](https://github.com/chvlyl/ISIC2018)), which cannot be a macro average if the winner's is 0.292. The likely explanation is a single Jaccard pooled over all attributes' pixels at once, which pigment network would dominate (72% of the positive pixels in this test split) — but I could not verify that from a primary source, so it is stated here as an open question rather than a fact.

Takeaway: quote the aggregation, or the number means nothing.

## Task 3 — Disease classification

Seven-class classification over the HAM10000-derived Task 3 set (10,015 images):

| Code | Class | | Code | Class |
|---|---|---|---|---|
| MEL | Melanoma 黑素瘤 | | AKIEC | Actinic keratoses 日光性角化 |
| NV | Melanocytic nevi 痣 | | BKL | Benign keratosis-like 良性角化 |
| BCC | Basal cell carcinoma 基底细胞癌 | | DF | Dermatofibroma 皮肤纤维瘤 |
| | | | VASC | Vascular lesions 血管病变 |

![Sample images by class](assets/class-samples.png)

This task has two versions in this repo. The 2021 one is kept because its failure is instructive.

### The 2021 version, and the bug that invalidates its number

The original notebook trained a small `Sequential` CNN on **32×32** images and reported **61.5%** test accuracy. Two problems, both discovered in 2026 while auditing:

**It leaked.** `resample(replace=True, n_samples=500)` upsampled each class by *duplication*, and only afterwards did `train_test_split` run. DF has 60 original images and VASC 80 — duplicated to 500 each, the same physical image necessarily lands in both train and test. **That 61.5% is contaminated and is not a valid baseline for anything.**

**And 32×32 was the wrong call.** The diagnostic signal in dermoscopy *is* fine texture — pigment network, streaks, border irregularity. Thirty-two pixels destroys precisely the information the task depends on. `EfficientNetB0` was imported in that notebook but never wired up; transfer learning was the plan that didn't get finished.

### The 2026 version

Same data, done properly: **EfficientNetB0 at 224×224**, ImageNet-pretrained, two-phase (frozen head → fine-tune top 60 layers with BatchNorm kept frozen), stratified 70/15/15 split on all 10,015 images with saved split ids, sqrt-inverse class weights instead of duplicate-resampling, label smoothing 0.05.

| Metric | Score | |
|---|---|---|
| Accuracy | **0.801** | majority-class baseline (always NV) = 0.669 |
| Balanced accuracy | **0.703** | mean per-class recall — the number comparable to a balanced test set |
| Macro F1 | **0.655** | |
| Macro ROC-AUC (ovr) | **0.944** | |
| ECE (10-bin) | **0.041** | well calibrated |

Per class, on the 1,503-image test split:

| Class | Support | Sensitivity | Specificity | F1 |
|---|---|---|---|---|
| MEL | 167 | **0.419** | 0.951 | 0.464 |
| NV | 1006 | 0.892 | 0.841 | 0.905 |
| BCC | 77 | 0.857 | 0.974 | 0.733 |
| AKIEC | 49 | 0.592 | 0.982 | 0.558 |
| BKL | 165 | 0.685 | 0.948 | 0.651 |
| DF | 17 | 0.706 | 0.989 | 0.533 |
| VASC | 22 | 0.773 | 0.995 | 0.739 |

**Read accuracy with suspicion.** On this natural distribution, always predicting NV scores 0.669. Raw accuracy is therefore a weak signal; balanced accuracy (0.703) and macro-F1 (0.655) are the honest summaries, and macro-AUC 0.944 says the ranking is much better than the argmax decision suggests.

### The number that actually matters is the worst one

**Melanoma sensitivity is 0.419** — 97 of 167 melanomas missed. Specificity is 0.951, which in cancer screening is the wrong side of the trade: a false negative is a missed cancer, a false positive is a biopsy. Melanoma is also the class most confusable with NV and BKL.

Taking argmax is the wrong decision rule for this problem. A screening deployment would instead fit the operating point to a required sensitivity (say ≥0.95 for MEL) and accept the resulting false-positive load — a different model output, not a different model.

### Abstention works, and that is the useful finding

The model is well calibrated (ECE 0.041), which makes "I don't know" a usable output:

| Abstain if confidence < | Coverage | Accuracy on kept |
|---|---|---|
| — (no abstention) | 100% | 0.801 |
| 0.60 | 74.3% | 0.901 |
| 0.80 | 53.0% | 0.951 |
| 0.95 | 21.2% | 0.994 |

Routing the least-confident cases to a human buys 95% accuracy on half the volume, or 99% on a fifth. For triage that is worth more than squeezing another point of top-1 accuracy.

### What fixing it taught me (about my own 2026 code)

The first 2026 run scored only 0.635 — barely above the leaky 0.615 — because of a bug I introduced: phase 2's `ModelCheckpoint` restarted its best-score tracking from scratch and **overwrote phase 1's better weights** (val 0.787) with a worse fine-tuned model (val 0.645). Fine-tuning had also been degrading the backbone because BatchNorm layers were left trainable, so their statistics were being rewritten on small batches. Freezing BN and keeping one checkpoint per phase, then explicitly choosing the winner, turned 0.635 into 0.801.

The lesson generalises: a two-phase training script needs *one* notion of "best", and a plausible-looking marginal improvement is often a bug rather than a ceiling.

---

## What I'd do differently

Written in hindsight, several years and a few production systems later. Where the 2026 pass actually fixed something, it says so — the rest are still open.

**1. ✅ The classifier was the weak half.** The 2021 model was a `Sequential` CNN on **32×32** images. Thirty-two pixels destroys exactly the fine texture dermoscopy diagnosis depends on. `EfficientNetB0` was imported and never wired up. *Fixed in 2026:* transfer learning at 224×224 took balanced accuracy from an invalid 0.615 to a measured **0.703**, macro-AUC to **0.944**.

**2. ✅ I reported the number I wanted, not the number I had.** An earlier version of this README claimed "87% accuracy" and "EfficientNet". Neither was true of the code: the saved output said 0.615, and the architecture was the small CNN. *Corrected 2026* — and then the 0.615 itself turned out to be invalid (see next).

**3. ✅ Resampling didn't just blunt the imbalance, it leaked.** `resample(replace=True, n_samples=500)` ran **before** `train_test_split`, so duplicated minority images (DF: 60 originals → 500; VASC: 80 → 500) appeared in train and test simultaneously. This is the single most serious defect in the original project, and I didn't notice it for five years. *Fixed in 2026:* stratified split on original images only, sqrt-inverse class weights instead of duplication, split ids committed.

**4. ✅ Segmentation, attributes and classification never met.** Three notebooks, no data flow between them, even though chaining them is the entire point of doing all three tasks. *Addressed in 2026* by the [concept bottleneck pipeline](concept-bottleneck-pipeline.ipynb): Task 2's attributes become an interpretable middle layer that Task 3's diagnosis is predicted *from*, so the model states its reasoning in dermoscopic vocabulary. See [What the AI era changes](#what-the-ai-era-changes).

**5. ⬜ Still no cross-validation and no error bars.** Single split, `random.seed` set in one notebook but not the other. Every 2026 number here is also a single stratified split with a fixed seed — better documented, but still a point estimate. A stratified k-fold would turn "0.703" into "0.703 ± something", and without the ±, a 2-point difference between models means nothing.

**6. ⬜ Argmax is the wrong decision rule for cancer screening.** Both versions report the argmax class. But **melanoma sensitivity is 0.419** — 97 of 167 melanomas missed — against specificity 0.951. That trade-off is backwards for screening: a false negative is a missed cancer, a false positive is a biopsy. The right approach is to fit the operating point to a required sensitivity (≥0.95 for MEL) and accept the false-positive load. Same model, different output.

**7. ⬜ Notebooks were the wrong home for the reusable parts.** Metrics, data pipelines and model builders are copy-pasted between notebooks with small divergences. The 2026 Kaggle training scripts repeated this. Extracting them into a module — with unit tests on the metric functions — would make the ablation trustworthy, since every variant would provably share one evaluation path. Concretely: the `dice_macro` used here scores ~1.0 on an empty channel correctly predicted empty, so it is inflated by absent attributes and is only a training signal, never a score. A unit test would have made that obvious immediately instead of requiring a code comment.

**8. ⬜ No fairness audit.** HAM10000 is overwhelmingly light skin (Fitzpatrick I–III). Nothing in this repo measures how these models behave on darker skin, where melanoma carries higher mortality because it is caught later. A 2026 project ought to report metrics stratified by skin tone, using something like Diverse Dermatology Images or Fitzpatrick17k. Not done here, and it is the largest remaining gap.

---

## What the AI era changes

The 2021 framing was "train a segmenter, train a classifier". Several things that were research topics then are commodities now, and one of them reframes this whole project.

**Task 2 turns out to be the valuable half.** In 2018, attribute detection was a niche sub-task — the one I skipped. But its five attributes *are* the clinical vocabulary of dermoscopy, which makes them exactly what a **concept bottleneck model** needs: predict human-interpretable concepts, then diagnose *from those concepts*, so the output is "melanoma, because irregular streaks and negative network" rather than an unexplained label plus a heatmap. That is implemented in [`concept-bottleneck-pipeline.ipynb`](concept-bottleneck-pipeline.ipynb).

There is a hard constraint worth recording, because it is easy to assume otherwise: **Task 1-2 and Task 3 images do not overlap at all.** Measured, not assumed — Task 1-2 ids run `ISIC_0000000`–`ISIC_0016072` (2,594 images) and Task 3 is HAM10000 at `ISIC_0024306`–`ISIC_0034320` (10,015), from different contributing institutions. Zero shared ids. So concepts for Task 3 images must be *predicted*, never read from ground truth — which is also the real deployment setting, since no clinician hands you attribute annotations at inference time.

And a caution the Task 2 results already raise: the attribute model appears to partly encode "where is the lesion" rather than true attribute texture (see [above](#what-the-model-actually-learned--and-why-that-matters-downstream)). A bottleneck built on concepts that don't mean what they claim is interpretable in name only, which is why the pipeline measures **concept fidelity** and not just downstream accuracy.

Other directions, deliberately listed rather than implemented:

- **Foundation models instead of ImageNet.** SAM / MedSAM make from-scratch U-Net a weak baseline for Task 1; DINOv2-class backbones or dermatology-specific foundation models would beat ImageNet-pretrained ResNet50/EfficientNet on a dataset this size.
- **Vision-language models for the long tail.** Rather than resampling or weighting DF (60 images) and VASC (80), zero-/few-shot text-prompted classification sidesteps the requirement for enough examples per class.
- **Diffusion-generated rare classes** — viable now, but only for training data, with validation and test kept strictly real, because whether synthetic dermoscopy preserves diagnostic texture is unresolved.
- **Uncertainty as a first-class output.** The abstention curve above is a crude version; conformal prediction would give prediction sets with coverage guarantees.
- **Metadata and report generation.** HAM10000 ships age, sex and lesion site, all unused here. Image → attributes → structured clinical narrative is a natural next layer.
- **Core ML export.** The 2021 "future work" said "lightweight models for mobile"; that is now a small amount of work, and on-device inference is where a tool like this would actually be used.

---

## Setup

```bash
pip install -r requirements.txt
```

The notebooks are written for **Google Colab with a GPU runtime** (they mount Google Drive and use `google.colab` helpers). `requirements.txt` pins the 2021-era TensorFlow 2.5 stack they were built against — see the comments in that file for the two version traps.

**Data** is not redistributed here. See [`data/README.md`](data/README.md) for the download link and the exact directory layout the notebooks expect.

Run order: Task 1 → Task 2 → Task 3, which are independent of each other. The concept-bottleneck notebook is the exception: it consumes a trained Task 2 attribute model.

The 2026 results were produced on free Kaggle GPU (P100) rather than Colab — Task 2 in ~30 min, Task 3 in ~15 min. Raw metrics are committed as [`assets/task2-results.json`](assets/task2-results.json) and [`assets/task3-results.json`](assets/task3-results.json).

## License

MIT — see [LICENSE](LICENSE).
