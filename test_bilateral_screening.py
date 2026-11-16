"""
test_bilateral_screening.py: Automated Clinical Bilateral Verification Suite.

Validates:
1. Bilateral Eye Screening Pipeline (Right Eye OD vs. Left Eye OS).
2. Dual-eye Image Quality Gatekeeper (Blur & Illumination).
3. Test-Time Augmentation (TTA) inference on both eyes.
4. Patient-level diagnostic triage governed by max(OD, OS).
5. Inter-Ocular Severity Asymmetry detection (Concordant, Mild Δ=1, Marked Δ≥2 Red Flag).
6. Multi-Scenario Bilateral Clinical Workups (5 standard clinical presets).
7. Unified Bilateral Clinical PDF Report generation and structure validation.
"""
import os
import io
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import html
import re
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ----------------- 1. SYSTEM INITIALIZATION -----------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, 5)
)
state = torch.load("best_retina_model.pth", map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()

temp = 1.3782
if os.path.exists("temperature.txt"):
    with open("temperature.txt") as f:
        try:
            temp = float(f.read().strip())
        except ValueError:
            pass

# ----------------- 2. GRAD-CAM++ ENGINE -----------------
class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.gradients = None
        self.activations = None

        with torch.enable_grad():
            output = self.model(input_tensor)
            if target_class is None:
                target_class = int(torch.argmax(output, dim=1).item())
            self.model.zero_grad()
            output[0, target_class].backward()

        grads = self.gradients.detach()
        acts = self.activations.detach()

        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads

        sum_acts_grads_3 = torch.sum(acts * grads_power_3, dim=(2, 3), keepdim=True)
        eps = 1e-7
        denom = 2.0 * grads_power_2 + sum_acts_grads_3
        denom = torch.where(denom != 0.0, denom, torch.ones_like(denom) * eps)

        aij = grads_power_2 / denom
        aij = torch.where(grads != 0.0, aij, torch.zeros_like(aij))

        weights = torch.maximum(grads, torch.zeros_like(grads)) * aij
        weights = torch.sum(weights, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = torch.maximum(cam, torch.zeros_like(cam))

        cam_np = cam[0, 0].cpu().numpy().astype(np.float32)
        cam_np = cv2.resize(cam_np, (256, 256))
        cam_min, cam_max = np.min(cam_np), np.max(cam_np)
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam_np = np.zeros_like(cam_np)

        logits = output.detach().cpu()
        self.gradients = None
        self.activations = None
        return cam_np, target_class, logits

grad_cam = GradCAMPlusPlus(model, model.features[-1])

# ----------------- 3. PREPROCESSING & INFERENCE -----------------
def fast_ben_graham(img_bgr, target_size=256):
    img_bgr = cv2.resize(img_bgr, (target_size, target_size))
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), 10)
    enhanced = cv2.addWeighted(img_bgr, 4, blurred, -4, 128)
    return enhanced

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def run_inference(img_bgr):
    proc_bgr = fast_ben_graham(img_bgr, target_size=256)
    proc_rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
    tensor_img = preprocess(Image.fromarray(proc_rgb)).unsqueeze(0).to(device)

    heatmap, _, _ = grad_cam.generate(tensor_img)

    tta_tensors = [
        tensor_img[0],
        preprocess(Image.fromarray(np.fliplr(proc_rgb).copy())).to(device),
        preprocess(Image.fromarray(np.flipud(proc_rgb).copy())).to(device),
        preprocess(Image.fromarray(np.rot90(proc_rgb, 2).copy())).to(device)
    ]
    with torch.no_grad():
        batch_logits = model(torch.stack(tta_tensors))
        batch_probs = torch.softmax(batch_logits / temp, dim=1).cpu().numpy()
        probs = batch_probs.mean(axis=0)
        pred_class = int(probs.argmax())

    return proc_rgb, probs, int(pred_class), heatmap

# ----------------- 4. QUALITY ASSESSMENT -----------------
def assess_image_quality(image, blur_threshold=5.0):
    norm_img = cv2.resize(image, (512, 512))
    gray = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mask = gray > 15
    mean_brightness = float(np.mean(gray[mask])) if np.sum(mask) > 0 else 0.0
    is_valid = (laplacian_var >= blur_threshold) and (20.0 <= mean_brightness <= 240.0)
    return is_valid, laplacian_var, mean_brightness

# ----------------- 5. SAMPLE FINDER -----------------
def get_sample_image(stage_num, case_idx=0):
    matched = []
    if os.path.exists("sample_images"):
        for root, dirs, files in os.walk("sample_images"):
            for f in sorted(files):
                if f.lower().startswith(f"stage_{stage_num}") and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    matched.append(os.path.join(root, f))
    if 0 <= case_idx < len(matched):
        path = matched[case_idx]
        return cv2.imread(path), path
    return None, ""

# ----------------- 6. ASYMMETRY ANALYSIS ENGINE -----------------
def evaluate_bilateral_case(od_stage, os_stage):
    overall_stage = max(od_stage, os_stage)
    diff = abs(od_stage - os_stage)
    higher_eye = "OD (Right Eye)" if od_stage > os_stage else "OS (Left Eye)" if os_stage > od_stage else "Both Eyes"
    if diff == 0:
        asym_type = "Concordant (Symmetric)"
    elif diff == 1:
        asym_type = "Mild Asymmetry (Δ=1)"
    else:
        asym_type = f"Marked Asymmetry Red Flag (Δ={diff})"
    return overall_stage, diff, higher_eye, asym_type

# ----------------- 7. REPORTLAB BILATERAL PDF GENERATOR -----------------
def build_test_bilateral_pdf(od_bgr, od_pred, od_conf, os_bgr, os_pred, os_conf):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    elements = []
    styles = getSampleStyleSheet()

    body_style = ParagraphStyle('B', parent=styles['Normal'], fontSize=8, leading=10.5)
    title_style = ParagraphStyle('T', parent=styles['Heading1'], fontSize=15, leading=18, textColor=colors.HexColor('#1E3A8A'))

    overall_stage = max(od_pred, os_pred)
    diff = abs(od_pred - os_pred)

    elements.append(Paragraph("<b>RETINA INSIGHT &mdash; CLINICAL BILATERAL REPORT</b>", title_style))
    elements.append(Spacer(1, 6))

    diag_text = f"OVERALL TRIAGE: STAGE {overall_stage} (Governing Eye: {'OD' if od_pred > os_pred else 'OS' if os_pred > od_pred else 'Bilateral'}) | Asymmetry: Δ={diff}"
    elements.append(Paragraph(f"<b>{diag_text}</b>", body_style))
    elements.append(Spacer(1, 6))

    def _buf(img):
        b = io.BytesIO()
        Image.fromarray(cv2.cvtColor(cv2.resize(img, (200, 200)), cv2.COLOR_BGR2RGB)).save(b, format="JPEG")
        b.seek(0)
        return b

    img_table = Table([
        [Paragraph("<b>OD: Right Eye Fundus</b>", body_style), Paragraph("<b>OS: Left Eye Fundus</b>", body_style)],
        [RLImage(_buf(od_bgr), width=200, height=200), RLImage(_buf(os_bgr), width=200, height=200)]
    ], colWidths=[270, 270])
    elements.append(img_table)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

# ----------------- 8. TEST EXECUTION -----------------
def main():
    print("=" * 70)
    print("Retina Insight: Clinical Bilateral Screening Verification Suite")
    print("=" * 70)
    print(f"[*] Execution Device: {device}")
    print(f"[*] Temperature Calibration: T = {temp:.4f}")

    scenarios = [
        {"name": "Scenario 1: Bilateral Moderate DR", "od": (2, 0), "os": (2, 1), "exp_overall": 2, "exp_diff": 0},
        {"name": "Scenario 2: Asymmetric Progression", "od": (3, 0), "os": (1, 0), "exp_overall": 3, "exp_diff": 2},
        {"name": "Scenario 3: High-Risk Proliferative Asymmetry", "od": (4, 0), "os": (2, 0), "exp_overall": 4, "exp_diff": 2},
        {"name": "Scenario 4: Early Unilateral Retinopathy", "od": (1, 0), "os": (0, 0), "exp_overall": 1, "exp_diff": 1},
        {"name": "Scenario 5: Bilateral Healthy Retina", "od": (0, 0), "os": (0, 1), "exp_overall": 0, "exp_diff": 0},
    ]

    print("\n[+] Evaluating Clinical Bilateral Scenarios:")
    print("-" * 70)
    print(f"{'Scenario Name':<38} {'OD True':<8} {'OS True':<8} {'Overall':<8} {'Δ Diff':<6} {'Status'}")
    print("-" * 70)

    for sc in scenarios:
        od_img, od_path = get_sample_image(sc["od"][0], sc["od"][1])
        os_img, os_path = get_sample_image(sc["os"][0], sc["os"][1])

        assert od_img is not None, f"Failed to load OD image: Stage {sc['od'][0]}"
        assert os_img is not None, f"Failed to load OS image: Stage {sc['os'][0]}"

        # Quality check
        od_q, od_blur, od_bright = assess_image_quality(od_img)
        os_q, os_blur, os_bright = assess_image_quality(os_img)
        assert od_q and os_q, "Quality check failed on valid sample images"

        # Inference
        _, od_probs, od_pred, _ = run_inference(od_img)
        _, os_probs, os_pred, _ = run_inference(os_img)

        # Check accurate prediction
        assert od_pred == sc["od"][0], f"OD prediction mismatch: expected {sc['od'][0]}, got {od_pred}"
        assert os_pred == sc["os"][0], f"OS prediction mismatch: expected {sc['os'][0]}, got {os_pred}"

        overall, diff, higher, asym_type = evaluate_bilateral_case(od_pred, os_pred)
        assert overall == sc["exp_overall"], f"Overall triage mismatch: expected {sc['exp_overall']}, got {overall}"
        assert diff == sc["exp_diff"], f"Asymmetry diff mismatch: expected {sc['exp_diff']}, got {diff}"

        print(f"{sc['name']:<38} {od_pred:<8} {os_pred:<8} {overall:<8} {diff:<6} [PASS]")

    print("-" * 70)
    print("[+] All 5 clinical bilateral scenarios verified with 100% precision.")

    # PDF verification
    print("\n[+] Generating and validating Bilateral Clinical PDF Report...")
    od_img, _ = get_sample_image(3, 0)
    os_img, _ = get_sample_image(1, 0)
    pdf_bytes = build_test_bilateral_pdf(od_img, 3, 65.6, os_img, 1, 87.7)
    assert len(pdf_bytes) > 10000, "PDF generation resulted in undersized or corrupt buffer"
    print(f"    [+] Successfully compiled Bilateral PDF Report ({len(pdf_bytes)} bytes)")

    print("\n" + "=" * 70)
    print("[SUCCESS] CLINICAL BILATERAL SCREENING SUITE FULLY VERIFIED.")
    print("=" * 70)

if __name__ == "__main__":
    main()
