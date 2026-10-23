# Retina-Insight
# 👁️ Retina Insight: Explainable AI for Diabetic Retinopathy Screening

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Major Project Phase-1 & Phase-2**  
> **Department of Computer Science and Engineering (AI & ML)**  
> **CMR College of Engineering & Technology (UGC Autonomous), Hyderabad**  
> **Batch No.:** 38 | **Academic Year:** 2026–2027  
> **Under the Guidance of:** Ms. Ruquia Taha, Assistant Professor  
> **Team Members:**  
> - G. Aishwarya (23H51A66X3)  
> - J. Sravan Kumar (23H51A66Y1)  
> - L. Yashwanth (23H51A66Y6)  

---

## 📌 Executive Summary
**Retina Insight** is an end-to-end, clinically safe, deep learning-powered clinical decision support system for **Diabetic Retinopathy (DR)** severity grading from retinal fundus photographs. 

Unlike conventional "black-box" models that blindly predict on ungradable images, **Retina Insight** introduces a **3-tier clinical pipeline**:
1. **Clinical Gatekeeper (IQA & OOD Detection):** Rejects non-retinal inputs (faces, objects, text) and flags sub-optimal scans (blur, illumination faults) using Laplacian variance and hemoglobin chromatic analysis.
2. **Transfer Learning Core:** Fine-tuned **EfficientNet-B0** with class-balanced weighting to handle severe medical class imbalance across all 5 clinical stages.
3. **Clinical Trust Engine:** Post-hoc **Confidence Calibration via Temperature Scaling** combined with **Grad-CAM visual heatmaps** to highlight pathological microaneurysms and hemorrhages.

---

## 🏆 Key Performance Benchmarks

| Metric | Score Achieved | Clinical Benchmark Significance |
| :--- | :--- | :--- |
| **Quadratic Weighted Kappa (QWK)** | **`0.8833`** | Top 5% global benchmark on APTOS 2019 dataset (*Near-perfect clinical agreement*) |
| **5-Class Validation Accuracy** | **`81.30%`** | Balanced multiclass performance across stages 0 to 4 |
| **Expected Calibration Error (ECE)** | **`3.58%`** | Reduced from **4.81%** post-temperature scaling ($T = 1.3301$) |
| **Inference Latency** | **`< 120 ms`** | Real-time diagnostic response on standard consumer hardware |

---

## 🏗️ System Architecture

```text
[Retinal Fundus Image / Upload]
               │
               ▼
┌────────────────────────────────────────┐
│  Tier 1: Clinical Gatekeeper (Safety)   │
│  - Anatomical Hemoglobin Check         │ ──(Fails)──▶ [Halt Screening & Request Retake]
│  - Laplacian Blur & Exposure Analysis  │
└──────────────────┬─────────────────────┘
                   │ (Passes Clinical Quality)
                   ▼
┌────────────────────────────────────────┐
│  Tier 2: Preprocessing & Neural Core    │
│  - Ben Graham Color Contrast Filtering │
│  - EfficientNet-B0 (Trained Backbone)  │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│  Tier 3: Trust, Calibration & XAI      │
│  - Temperature Scaling Logits (T=1.33) │ ──▶ [Calibrated Probabilities & ECE Analysis]
│  - Grad-CAM Lesion Localization Heatmap│ ──▶ [Visual Clinical Decision Support]
└──────────────────┬─────────────────────┘
                   │
                   ▼
[Interactive Streamlit Diagnostic Dashboard]
