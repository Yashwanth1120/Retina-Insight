# Retina Insight: Explainable AI for Diabetic Retinopathy Screening

A Streamlit application that grades diabetic retinopathy (DR) from retinal fundus photographs on a 0 to 4 scale, shows how confident the model is, highlights the image regions that influenced the prediction (Grad-CAM), and exports a PDF screening report.

**Live demo:** [https://retina-insight-9amtvfcupzmubdubgigw4v.streamlit.app/](https://retina-insight-9amtvfcupzmubdubgigw4v.streamlit.app/)

> **Intended use: research and education only.** Retina Insight is a screening-aid prototype. It is not a medical device and its output is not a diagnosis. Every result must be reviewed by a qualified ophthalmologist.

## What it does

- **Grades DR in five stages:** No DR, Mild, Moderate, Severe, Proliferative (ICDR scale, 0 to 4).
- **Calibrated confidence:** temperature scaling (T = 1.3782) makes the displayed probabilities more reliable. Predictions below 60% confidence trigger a manual-review warning.
- **Grad-CAM attention map:** shows the regions that influenced the model's prediction. It has not been validated as a lesion detector.
- **Image check:** a simple colour, blur and brightness heuristic rejects obviously unsuitable images. It is not a trained retina detector and can be fooled.
- **Report generation:** drafts a patient-friendly summary and a SOAP-style clinical note, and exports a PDF. Notes come from an offline template, or optionally from Google Gemini. The patient name is never sent to Gemini.

## Results

### Research Retraining Benchmark (15 Epochs, Strict 3-Way Split & Scale Augmentation)

To address the limitations of rapid baseline training (5 epochs, unaugmented framing, optimistic internal validation), an academic-grade retraining pipeline is provided in `retina_insight_paper_retraining.ipynb`. It uses a strict 3-way stratified partition (70% train, 15% validation, 15% strictly held-out test), scale jitter (`RandomResizedCrop` 0.80–1.05), aspect framing, rotation, and color jitter, trained for 15 epochs with Cosine Annealing.

#### Strict Held-Out Test Evaluation (550 unseen APTOS scans)

| Metric | Baseline Prototype (5 Epochs) | Retrained Paper Model (15 Epochs) | Clinical Significance |
|---|---|---|---|
| **Data Partition** | 80/20 train/val (no test set) | **70% Train / 15% Val / 15% Test** | Unbiased test set isolated before training |
| **Quadratic Weighted Kappa (QWK)** | 0.8864 (optimistic val) | **0.8752** (held-out test) | Substantial agreement with retina specialists |
| **Referable DR (Stage ≥ 2) AUC** | Not evaluated on unseen test | **0.9821** | Primary triage benchmark for specialist referral |
| **Within-One-Stage Concordance** | Not reported | **95.64%** | Near-zero catastrophic stage misclassifications |
| **Exact 5-Class Accuracy** | 78.0% (val) | **77.45%** (held-out test) | Fine-grained severity classification |
| **Stage 0 (No DR) Sensitivity** | Not reported | **97.05%** (95.27% precision) | Prevents false referrals for healthy eyes |
| **Optimal Temperature (T)** | 1.3782 | **1.4146** | Fitted post-hoc on validation logits via L-BFGS |
| **Calibrated Test ECE** | N/A | **6.20%** | Reliable uncertainty calibration |

### APTOS 2019 validation split (internal)

The model was trained on 80% of the APTOS 2019 training images and evaluated on the remaining 20% (733 images, stratified split, seed 42).

| Metric | Value |
|---|---|
| Accuracy | 78.0% |
| Quadratic weighted kappa (QWK) | 0.8864 |
| Temperature (T) | 1.3782 |
| ECE before / after temperature scaling | 5.92% / 2.53% |

These numbers come from the final epoch (5) of the training run. The same validation set was used to choose the best epoch and to fit the temperature, so they are slightly optimistic. There is no separate APTOS test set.

### IDRiD external test (never seen during training)

Evaluated on the official IDRiD test split (103 images, from a different camera and clinic).

| Metric | As deployed | With border crop (experiment) |
|---|---|---|
| Accuracy | 27.2% | 40.8% |
| Within one stage | 83.5% | 87.4% |
| QWK | 0.470 | 0.637 |
| Referable DR (stage 2 or higher) AUC | 0.881 | 0.914 |
| Referable DR sensitivity | 73.4% | 64.1% |
| Referable DR specificity | 76.9% | 97.4% |
| ECE after scaling | 25.7% | 16.4% |

**What this means:** the model separates sicker from healthier eyes reasonably well on IDRiD (AUC 0.88), but it often picks the wrong exact stage, and its confidence is not well calibrated on images from a different source. Healthy eyes are frequently graded as Stage 1, and moderate cases as Stage 3. The border-crop variant improved the IDRiD results but lowered accuracy on APTOS-style images (92% to 72% on the demo set), so it is not used in the app.

## Limitations

- **Domain shift:** performance drops sharply on images that differ from APTOS in camera, field of view or framing. The model was trained with no cropping and no scale or framing augmentation.
- **Short training:** 5 epochs. Validation QWK was still rising at the last epoch.
- **No separate APTOS test set:** see the note above about optimistic internal numbers.
- **Small classes:** the IDRiD test split has only 5 Stage 1 images, so Stage 1 figures are very uncertain.
- **Grad-CAM is not clinically validated.** A heatmap showing where the model looked does not prove it found a lesion.
- **The image check is a heuristic** and can accept non-retinal images or reject valid ones.
- **Not a medical device.** Not tested in a clinical setting.

## How it works

1. **Image check:** colour statistics, blur (Laplacian variance) and brightness.
2. **Preprocessing:** resize the whole image to 256 x 256, then a Ben Graham filter (Gaussian blur with sigma 10).
3. **Model:** EfficientNet-B0 (ImageNet weights) with a 5-class head, dropout 0.3, class-weighted cross-entropy, AdamW (learning rate 3e-4), cosine schedule, 5 epochs. Augmentation: flips and rotation up to 15 degrees.
4. **Calibration:** temperature scaling fitted with L-BFGS on the validation logits.
5. **Explanation:** Grad-CAM on the last feature block.
6. **Notes and PDF:** offline template or Gemini (optional), exported with ReportLab.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional: paste a Gemini API key in the sidebar for AI-drafted notes. Without a key, the app uses the built-in offline summary.

On Streamlit Community Cloud, `requirements.txt` and `packages.txt` are used automatically.

## Evaluate the model on your own labelled images

```bash
python evaluate_model.py --data test_images --csv labels.csv --out eval_results
```

`labels.csv` needs the columns `id_code` and `diagnosis` (0 to 4); alternatively, name the files `stage_N_...` or put them in folders `0` to `4`. The script reports accuracy, QWK, the confusion matrix, per-class sensitivity and specificity, referable-DR AUC, calibration (ECE and NLL), and accuracy by confidence level. Use `--crop` to test border cropping. Use only images that were never used for training or calibration.

## Repository contents

| Path | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `retina_insight_paper_retraining.ipynb` | Research retraining notebook (15 epochs, 3-way split, scale augmentation, calibration, IDRiD discovery) |
| `best_retina_model.pth` | Trained EfficientNet-B0 weights |
| `temperature.txt` | Fitted calibration temperature |
| `sample_images/` | 25 demo images from APTOS (22 were in the training split, 3 in validation; for demonstration only, not evidence of performance) |
| `evaluate_model.py` | Evaluation script |
| `requirements.txt`, `packages.txt` | Dependencies for deployment |

## Data and attribution

- **APTOS 2019 Blindness Detection** (Asia Pacific Tele-Ophthalmology Society, via Kaggle): used for training and internal validation. Use is subject to the competition's data terms.
- **IDRiD (Indian Diabetic Retinopathy Image Dataset):** used for external testing. The images are not included in this repository. Licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Original source: IEEE DataPort. Citation: Porwal P., Pachade S., Kamble R., Kokare M., Deshmukh G., Sahasrabuddhe V. *Indian Diabetic Retinopathy Image Dataset (IDRiD): A Database for Diabetic Retinopathy Screening Research.* Data, 2018, 3(3), 25. DOI: 10.3390/data3030025.

## Team

**B.Tech Major Project (Batch 38, Academic Year 2026–2027)**  
**Department of Computer Science and Engineering (AI & ML)**  
**CMR College of Engineering & Technology (UGC Autonomous), Hyderabad**  

- **Team Members:**
  - L. Yashwanth (23H51A66Y6)
  - G. Aishwarya (23H51A66X3)
  - J. Sravan Kumar (23H51A66Y1)
- **Project Guide:** Ms. Ruquia Taha, Assistant Professor

## License

MIT. See the `LICENSE` file.
