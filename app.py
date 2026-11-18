import pandas as pd
import urllib.parse
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
import io
import re
import html
import zipfile
from datetime import datetime

# Optional Google GenAI SDK
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# PDF Generation Libraries
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Retina Insight | Clinical Decision Support",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- WHITE-LABELING CSS -----------------
# NOTE: the <header> is intentionally NOT hidden, because it contains the
# sidebar open/close toggle on several Streamlit versions.
hide_branding_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden; display: none;}
    [data-testid="stToolbar"] {visibility: hidden; display: none;}
    [data-testid="stDecoration"] {visibility: hidden; display: none;}
    [data-testid="stStatusWidget"] {visibility: hidden; display: none;}
    .stDeployButton {display: none;}
    #manage-app-button {display: none;}
    div[data-testid="stToolbarActions"] {display: none;}
    </style>
"""
st.markdown(hide_branding_style, unsafe_allow_html=True)

st.title("👁️ Retina Insight: Clinical Decision Support System")
st.caption("AI-Powered Automated Screening, Confidence Calibration, Grad-CAM Attention Maps & Multimodal Notes")
st.warning(
    "⚠️ **Screening aid only — not a medical diagnosis.** Results must be reviewed and "
    "confirmed by a qualified ophthalmologist before any clinical decision."
)

# Final training run (Kaggle notebook log): APTOS validation split, 733 images (20%, seed 42).
# The same split was used to pick the best epoch and to fit the temperature, so these are slightly optimistic.
BENCHMARKS = {"qwk": "0.8864", "accuracy": "78.0%", "ece": "2.53%"}
# External test on the official IDRiD test split (103 images, never seen in training):
EXTERNAL = {"accuracy": "27.2%", "qwk": "0.47", "auc_referable": "0.88"}
LOW_CONFIDENCE_THRESHOLD = 60.0  # percent


# ----------------- CLINICAL CARE PROTOCOLS & EYE CARE DIRECTORY (ICO/WHO GUIDELINES) -----------------
CLINICAL_PROTOCOLS = {
    0: {
        "urgency": "Routine Annual Review",
        "badge_color": "green",
        "timeline": "Re-screen in 12 months",
        "referral_level": "Primary Care / Optometry / General Ophthalmology",
        "summary": "No microvascular lesions detected. Low immediate vision risk.",
        "target_goals": [
            "Maintain optimal glycemic control (target HbA1c < 7.0%).",
            "Maintain blood pressure within target limits (< 130/80 mmHg).",
            "Annual comprehensive dilated eye examination."
        ],
        "workup": "Routine visual acuity & non-mydriatic fundus evaluation."
    },
    1: {
        "urgency": "Early Monitoring (Low Risk)",
        "badge_color": "orange",
        "timeline": "Follow-up in 6 to 9 months",
        "referral_level": "Comprehensive Ophthalmologist / Medical Retina Clinic",
        "summary": "Microaneurysms only. Early vascular stress without significant macular involvement.",
        "target_goals": [
            "Intensify glycemic and metabolic management with treating physician.",
            "Monitor blood pressure and lipid profile to prevent lesion progression.",
            "Patient education on self-reporting sudden vision changes or floaters."
        ],
        "workup": "Dilated slit-lamp biomicroscopy & baseline macular monitoring."
    },
    2: {
        "urgency": "Referable Diabetic Retinopathy (Moderate Risk)",
        "badge_color": "red",
        "timeline": "Specialist referral within 2 to 3 months",
        "referral_level": "Comprehensive Ophthalmologist / Retina Clinic",
        "summary": "Microaneurysms, hemorrhages, or exudates detected. Requires dilated evaluation.",
        "target_goals": [
            "Formal dilated eye evaluation to assess for diabetic macular edema (DME).",
            "Avoid rapid, uncontrolled fluctuations in blood sugar.",
            "Coordinate diabetes care between endocrinologist and eye specialist."
        ],
        "workup": "Optical Coherence Tomography (OCT Macula) + Dilated Fundus Examination."
    },
    3: {
        "urgency": "High-Risk Pre-Proliferative (Severe)",
        "badge_color": "red",
        "timeline": "URGENT consultation within 2 to 4 weeks",
        "referral_level": "Vitreo-Retinal Super-Specialist",
        "summary": "Multi-quadrant blot hemorrhages, venous beading, or IRMA. ~50% risk of progression to PDR within 1 year.",
        "target_goals": [
            "Urgent specialist evaluation for preventive laser or anti-VEGF therapy.",
            "Strict blood pressure control to reduce hemorrhage risk.",
            "Renal status check (microalbuminuria / eGFR) as nephropathy often coexists."
        ],
        "workup": "High-definition OCT Macula + Fundus Fluorescein Angiography (FFA)."
    },
    4: {
        "urgency": "Ophthalmic Emergency (Vision-Threatening)",
        "badge_color": "red",
        "timeline": "IMMEDIATE emergency review within 24 to 48 hours",
        "referral_level": "Tertiary Eye Hospital / Emergency Retina Service",
        "summary": "Neovascularization detected. High danger of sudden vitreous hemorrhage or tractional retinal detachment.",
        "target_goals": [
            "Immediate presentation to a tertiary vitreo-retinal facility.",
            "Urgent anti-VEGF injections, panretinal photocoagulation (PRP), or vitrectomy.",
            "Avoid strenuous physical lifting, heavy bending, or unapproved anticoagulants."
        ],
        "workup": "Emergency FFA, B-Scan Ultrasonography (if media hazy), and surgical evaluation."
    }
}

EYE_HOSPITALS_DIRECTORY = {
    "Telangana & Andhra Pradesh": [
        {"name": "L.V. Prasad Eye Institute (LVPEI)", "city": "Hyderabad (Banjara Hills, Kismatpur, Madhapur)", "contact": "+91 40 6810 2020", "type": "Apex Tertiary Retinal Center", "url": "https://www.lvpei.org"},
        {"name": "Sarojini Devi Eye Hospital", "city": "Hyderabad (Mehdipatnam)", "contact": "+91 40 2353 8206", "type": "Government Apex Regional Eye Hospital (Subsidized/Free)", "url": "https://dme.telangana.gov.in/sarojini-devi-eye-hospital-hyderabad/"},
        {"name": "Maxivision Super Speciality Eye Hospital", "city": "Hyderabad (Somajiguda, Kukatpally, AS Rao Nagar)", "contact": "1800 419 2020", "type": "Vitreo-Retina Specialty Network", "url": "https://www.maxivisioneyehospital.com"},
        {"name": "Apollo Hospitals Eye Institute", "city": "Hyderabad (Jubilee Hills / Secunderabad)", "contact": "+91 40 2360 7777", "type": "Multi-Specialty Retinal Center", "url": "https://www.apollohospitals.com"}
    ],
    "National Apex Eye Centers (India)": [
        {"name": "Dr. Rajendra Prasad Centre for Ophthalmic Sciences (AIIMS)", "city": "New Delhi", "contact": "+91 11 2658 8500", "type": "National Apex Institute", "url": "https://www.aiims.edu"},
        {"name": "Sankara Nethralaya", "city": "Chennai, Tamil Nadu", "contact": "+91 44 4227 1500", "type": "Premier Charitable Retina Center", "url": "https://www.sankaranethralaya.org"},
        {"name": "Aravind Eye Care System", "city": "Madurai, Coimbatore, Tirunelveli (Tamil Nadu)", "contact": "+91 452 435 6100", "type": "High-Volume Retinal Screening & Care", "url": "https://aravind.org"}
    ]
}

DIAGNOSIS_INFO = {
    0: {"name": "Stage 0: No Diabetic Retinopathy", "action": "Routine annual screening recommended in 12 months."},
    1: {"name": "Stage 1: Mild DR", "action": "Follow-up screening in 6-12 months. Maintain strict glycemic control."},
    2: {"name": "Stage 2: Moderate DR", "action": "Referral to ophthalmologist within 2-4 months for dilated eye examination."},
    3: {"name": "Stage 3: Severe DR", "action": "Urgent retinal specialist consultation within 2-4 weeks."},
    4: {"name": "Stage 4: Proliferative DR", "action": "Immediate emergency consultation required. High risk of permanent vision loss."}
}

CSME_INFO = {
    "Low": {
        "name": "CSME Negative / Low Risk",
        "badge_color": "green",
        "description": "No hard exudates or microvascular leakage detected within the central macular zone (> 2 disc diameters from fovea).",
        "workup": "Standard diabetic eye care. Re-screen per DR severity timeline."
    },
    "Moderate": {
        "name": "CSME Equivocal / Moderate Risk",
        "badge_color": "orange",
        "description": "Hard exudates or microvascular stress identified in macular periphery (1 to 2 disc diameters from foveal center).",
        "workup": "Optical Coherence Tomography (OCT Macula) recommended within 4-6 weeks to evaluate subclinical macular thickening."
    },
    "High": {
        "name": "CSME Positive / High Risk (Vision-Threatening)",
        "badge_color": "red",
        "description": "Clinically Significant Diabetic Macular Edema: Hard exudates, foveal microaneurysms, or dense leakage within central foveal ring (< 1 disc diameter / 500 μm).",
        "workup": "URGENT high-definition OCT Macula and retinal specialist evaluation for anti-VEGF injection or focal macular laser therapy."
    }
}

DISCLAIMER_TEXT = (
    "This report is generated by an automated screening aid and is not a medical diagnosis. "
    "All findings must be reviewed and confirmed by a qualified ophthalmologist."
)

# ----------------- GATEKEEPER (HEURISTIC) -----------------
def verify_retinal_anatomy(image):
    """Simple colour-statistics heuristic. It is NOT a reliable retina detector."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask = gray > 15
    if np.sum(mask) < (image.shape[0] * image.shape[1] * 0.10):
        return False, "Image rejected: mostly empty or blank."

    active_rgb = rgb[mask]
    active_hsv = hsv[mask]

    r_mean = float(np.mean(active_rgb[:, 0]))
    g_mean = float(np.mean(active_rgb[:, 1]))
    b_mean = float(np.mean(active_rgb[:, 2]))

    if r_mean < (1.15 * b_mean) or r_mean < (1.02 * g_mean):
        return False, f"Image does not look like a fundus photograph (colour check failed; R:{r_mean:.0f}, G:{g_mean:.0f}, B:{b_mean:.0f})."

    hues = active_hsv[:, 0]
    sats = active_hsv[:, 1]
    retinal_pixels = ((hues <= 35) | (hues >= 155)) & (sats >= 20)
    retinal_ratio = float(np.mean(retinal_pixels))

    if retinal_ratio < 0.40:
        return False, f"Image does not look like a fundus photograph (only {retinal_ratio*100:.1f}% of pixels match the expected colour range)."

    return True, "Passed: image resembles a retinal fundus photograph (heuristic colour check)"

def assess_image_quality(image, blur_threshold=5.0, min_brightness=20.0, max_brightness=240.0):
    is_fundus, fundus_msg = verify_retinal_anatomy(image)
    if not is_fundus:
        return False, fundus_msg, 0.0, 0.0

    norm_img = cv2.resize(image, (512, 512))
    gray = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    mask = gray > 15
    mean_brightness = float(np.mean(gray[mask])) if np.sum(mask) > 0 else 0.0

    if laplacian_var < blur_threshold:
        return False, f"Image is too blurry (Standardized Blur Score: {laplacian_var:.1f} < {blur_threshold:.1f})", laplacian_var, mean_brightness
    if mean_brightness < min_brightness or mean_brightness > max_brightness:
        return False, f"Improper illumination ({mean_brightness:.1f})", laplacian_var, mean_brightness

    return True, "Passed: image quality acceptable (blur and illumination checks)", laplacian_var, mean_brightness

# NOTE: preprocessing is intentionally unchanged so it matches what the model was trained on.
def fast_ben_graham(img_bgr, target_size=256):
    img_bgr = cv2.resize(img_bgr, (target_size, target_size))
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), 10)
    enhanced = cv2.addWeighted(img_bgr, 4, blurred, -4, 128)
    return enhanced

# ----------------- GRAD-CAM++ (EXPLAINABLE AI) -----------------
class GradCAMPlusPlus:
    """Grad-CAM++ implementation for second- and third-order gradient-weighted visual explanations.
    Reference: Chattopadhyay et al., 'Grad-CAM++: Generalized Gradient-based Visual Explanations' (2018).
    Captures multiple scattered lesions (microaneurysms, blot hemorrhages) with higher fidelity than standard Grad-CAM."""

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

        # Second and third order partial derivatives
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads

        sum_acts_grads_3 = torch.sum(acts * grads_power_3, dim=(2, 3), keepdim=True)
        eps = 1e-7
        denom = 2.0 * grads_power_2 + sum_acts_grads_3
        denom = torch.where(denom != 0.0, denom, torch.ones_like(denom) * eps)

        aij = grads_power_2 / denom
        aij = torch.where(grads != 0.0, aij, torch.zeros_like(aij))

        # Pixel-wise positive gradient weighting
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


# ----------------- CLINICAL COLORMAPS & OVERLAY GENERATOR -----------------
COLORMAP_OPTIONS = {
    "JET (Classic Ophthalmology)": cv2.COLORMAP_JET,
    "Inferno (High Contrast / Microaneurysms)": cv2.COLORMAP_INFERNO,
    "Turbo (Perceptually Uniform)": cv2.COLORMAP_TURBO,
    "Viridis (Accessible / Colorblind-Friendly)": cv2.COLORMAP_VIRIDIS
}

def generate_cam_overlay(base_rgb, heatmap, colormap_name="JET (Classic Ophthalmology)", alpha=0.45, show_contours=False):
    """Renders customized clinical overlay with selectable colormap, alpha opacity,
    and optional high-attention lesion boundary contouring."""
    cmap_code = COLORMAP_OPTIONS.get(colormap_name, cv2.COLORMAP_JET)
    heatmap_u8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    heatmap_colored = cv2.applyColorMap(heatmap_u8, cmap_code)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    alpha_val = float(np.clip(alpha, 0.05, 0.95))
    overlay = cv2.addWeighted(base_rgb, 1.0 - alpha_val, heatmap_colored, alpha_val, 0)

    if show_contours:
        # High-confidence activation threshold (> 60% peak intensity)
        thresh = np.uint8((heatmap > 0.60) * 255)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

    return overlay

# ----------------- CLINICALLY SIGNIFICANT MACULAR EDEMA (CSME) DETECTOR -----------------
def detect_macular_edema_risk(img_bgr, heatmap=None, dr_stage=0):
    """
    Evaluates Clinically Significant Diabetic Macular Edema (CSME) Risk.
    
    Clinical Criteria (ETDRS & ICO Standards):
    - Negative / Low: No hard exudates within the macular field (> 2 disc diameters from fovea).
    - Moderate Risk: Hard exudates detected in macular periphery (1-2 disc diameters from foveal center).
    - High Risk (CSME Positive): Hard exudates or focal microvascular leakage within central foveal ring (< 1 disc diameter / 500 μm),
      or dense central macular Grad-CAM++ attention during moderate/severe DR.
    """
    resized = cv2.resize(img_bgr, (512, 512))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    center_x, center_y = 256, 256
    dd = 80  # Estimated optic disc diameter in 512x512 standard scale
    foveal_radius = int(dd * 0.5)      # ~40px (~500 um foveal zone)
    macular_radius = int(dd * 1.5)     # ~120px (macular region)

    retina_mask = gray > 20
    
    g_channel = resized[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g_enhanced = clahe.apply(g_channel)

    retina_median_g = np.median(g_enhanced[retina_mask]) if np.sum(retina_mask) > 0 else 128
    exudate_cand_g = (g_enhanced > (retina_median_g + 45)) & retina_mask

    hues = hsv[:, :, 0]
    sats = hsv[:, :, 1]
    vals = hsv[:, :, 2]
    yellow_mask = (hues >= 15) & (hues <= 55) & (sats >= 30) & (sats <= 220) & (vals >= 130)

    combined_exudates = (exudate_cand_g & yellow_mask).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_exudates = cv2.morphologyEx(combined_exudates, cv2.MORPH_OPEN, kernel)

    y_coords, x_coords = np.where(cleaned_exudates > 0)
    foveal_exudate_count = 0
    macular_exudate_count = 0
    total_exudate_pixels = len(x_coords)

    if total_exudate_pixels > 0:
        distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        foveal_exudate_count = int(np.sum(distances <= foveal_radius))
        macular_exudate_count = int(np.sum((distances > foveal_radius) & (distances <= macular_radius)))

    foveal_cam_intensity = 0.0
    if heatmap is not None:
        cam_resized = cv2.resize(heatmap, (512, 512))
        y_grid, x_grid = np.ogrid[:512, :512]
        dist_from_center = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
        macular_cam_mask = dist_from_center <= macular_radius
        foveal_cam_intensity = float(np.mean(cam_resized[macular_cam_mask])) if np.sum(macular_cam_mask) > 0 else 0.0

    risk_score = 0.0
    risk_score += min(foveal_exudate_count * 1.5, 45.0)
    risk_score += min(macular_exudate_count * 0.4, 25.0)
    risk_score += foveal_cam_intensity * 20.0
    if dr_stage >= 2:
        risk_score += (dr_stage - 1) * 3.5

    risk_score = float(np.clip(risk_score, 0.0, 100.0))

    if foveal_exudate_count >= 15 or risk_score >= 50.0 or (dr_stage >= 3 and risk_score >= 35.0):
        key = "High"
    elif macular_exudate_count >= 10 or risk_score >= 22.0 or (dr_stage >= 2 and risk_score >= 18.0):
        key = "Moderate"
    else:
        key = "Low"

    info = CSME_INFO[key]

    annotated = rgb.copy()
    # Foveal boundary (< 500 μm ring) in cyan
    cv2.circle(annotated, (center_x, center_y), foveal_radius, (0, 255, 255), 1)
    # Outer Macular boundary (1.5 DD) in blue
    cv2.circle(annotated, (center_x, center_y), macular_radius, (60, 120, 255), 1)
    # Contours of detected hard exudates in yellow
    contours, _ = cv2.findContours(cleaned_exudates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, contours, -1, (255, 230, 0), 1)

    return {
        "key": key,
        "name": info["name"],
        "status": info["name"],
        "score": risk_score,
        "badge_color": info["badge_color"],
        "description": info["description"],
        "action": info["workup"],
        "foveal_exudates": foveal_exudate_count,
        "macular_exudates": macular_exudate_count,
        "foveal_cam_intensity": foveal_cam_intensity,
        "annotated_rgb": annotated
    }

# ----------------- SCREENING CAMP BATCH ARCHIVE GENERATOR -----------------
def generate_camp_batch_archive(batch_results, camp_info):
    """Generates Camp Manifest CSV and comprehensive ZIP package with overlays and clinical summary."""
    clean_records = []
    for r in batch_results:
        rec = {k: v for k, v in r.items() if not k.startswith('_')}
        clean_records.append(rec)

    df_manifest = pd.DataFrame(clean_records)
    csv_bytes = df_manifest.to_csv(index=False).encode('utf-8')

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("camp_triage_manifest.csv", csv_bytes)

        total = len(batch_results)
        valid = sum(1 for r in batch_results if r.get("Quality_Status") == "Passed")
        rejected = total - valid
        referable = sum(1 for r in batch_results if "Yes" in str(r.get("Referable_DR", "")))
        urgent = sum(1 for r in batch_results if any(s in str(r.get("DR_Stage", "")) for s in ["Stage 3", "Stage 4"]))
        csme_high = sum(1 for r in batch_results if "High" in str(r.get("CSME_Risk", "")))

        summary_txt = f"""======================================================================
RETINA INSIGHT — SCREENING CAMP EXECUTIVE TRIAGE SUMMARY
======================================================================
Screening Facility:     {camp_info.get('camp_name', 'Community Vision Camp')}
Location / District:    {camp_info.get('location', 'Hyderabad Rural')}
Screening Officer:      {camp_info.get('screener_id', 'Field Officer')}
Date & Time:            {datetime.now().strftime('%d %B %Y, %I:%M %p')}
Software Engine:        Retina Insight v2.5 (Hospital & Camp Edition)
======================================================================
EPIDEMIOLOGICAL CAMP SCREENING STATISTICS:
----------------------------------------------------------------------
Total Patients Screened:        {total}
Usable / Cleared Fundus Scans:  {valid} ({valid/max(total, 1)*100:.1f}%)
Quality Rejections (Re-scan):   {rejected} ({rejected/max(total, 1)*100:.1f}%)
Referable DR Cases (Stage >= 2): {referable} ({referable/max(total, 1)*100:.1f}%)
Sight-Threatening (Stage 3-4):  {urgent} ({urgent/max(total, 1)*100:.1f}%)
Clinically Significant DME:     {csme_high} ({csme_high/max(total, 1)*100:.1f}%)
======================================================================
TRIAGE ACTION DIRECTIVE (ICO / WHO STANDARDS):
- Patients with Stage 3 (Severe DR), Stage 4 (Proliferative DR), or High CSME
  Risk require prioritized tertiary vitreoretinal referral.
- Any scan failing automated quality heuristics must be re-acquired on site.
======================================================================
"""
        zip_file.writestr("camp_executive_summary.txt", summary_txt)

        for res in batch_results:
            overlay_buf = res.get("_overlay_buf")
            scan_id = re.sub(r'[^A-Za-z0-9_-]+', '_', str(res.get("Scan_ID", "scan")))
            if overlay_buf:
                zip_file.writestr(f"gradcam_overlays/{scan_id}_gradcam.jpg", overlay_buf)

    zip_buffer.seek(0)
    return csv_bytes, zip_buffer.getvalue()

# ----------------- MODEL LOADER -----------------
DEFAULT_TEMPERATURE = 1.3782

@st.cache_resource
def load_system():
    if not os.path.exists("best_retina_model.pth"):
        raise FileNotFoundError("best_retina_model.pth was not found next to app.py.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 5)
    )

    state = torch.load("best_retina_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    temp = DEFAULT_TEMPERATURE
    temp_from_file = False
    if os.path.exists("temperature.txt"):
        try:
            with open("temperature.txt", "r") as f:
                value = float(f.read().strip())
            if value > 0:
                temp = value
                temp_from_file = True
        except ValueError:
            pass

    cam = GradCAMPlusPlus(model, model.features[-1])
    return model, cam, temp, temp_from_file, device

try:
    model, grad_cam, temperature, temp_from_file, device = load_system()
except Exception as e:
    st.error(f"🚨 Model could not be loaded, so screening is disabled: {e}")
    st.stop()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_data(show_spinner="Running model...")
def run_inference(img_bgr, temp, use_tta=True, multi_scale=False):
    """Preprocess, predict, and compute the Grad-CAM++ heatmap.
    When use_tta is True, executes 4-fold Test-Time Augmentation (standard, horizontal
    flip, vertical flip, rot180) to maximize classification precision and prevent
    borderline misclassifications.
    When multi_scale is True, incorporates central foveal multi-crop evaluation (Ensemble Mode).
    Cached so typing in the patient form or adjusting sliders does not re-run the neural network."""
    proc_bgr = fast_ben_graham(img_bgr, target_size=256)
    proc_rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
    tensor_img = preprocess(Image.fromarray(proc_rgb)).unsqueeze(0).to(device)

    heatmap, single_pred, single_logits = grad_cam.generate(tensor_img)

    if use_tta:
        tta_tensors = [
            tensor_img[0],
            preprocess(Image.fromarray(np.fliplr(proc_rgb).copy())).to(device),
            preprocess(Image.fromarray(np.flipud(proc_rgb).copy())).to(device),
            preprocess(Image.fromarray(np.rot90(proc_rgb, 2).copy())).to(device)
        ]
        if multi_scale:
            h, w = proc_rgb.shape[:2]
            margin = int(h * 0.075)
            cropped = proc_rgb[margin:h-margin, margin:w-margin]
            cropped_resized = cv2.resize(cropped, (256, 256))
            tta_tensors.append(preprocess(Image.fromarray(cropped_resized)).to(device))
            tta_tensors.append(preprocess(Image.fromarray(np.fliplr(cropped_resized).copy())).to(device))

        with torch.no_grad():
            batch_logits = model(torch.stack(tta_tensors))
            batch_probs = torch.softmax(batch_logits / temp, dim=1).cpu().numpy()
            probs = batch_probs.mean(axis=0)
            pred_class = int(probs.argmax())
    else:
        probs = torch.softmax(single_logits / temp, dim=1)[0].numpy()
        pred_class = int(single_pred)

    return proc_rgb, probs, int(pred_class), heatmap

# ----------------- CLINICAL NOTES (GENAI + OFFLINE) -----------------
def _offline_notes(stage_idx, stage_name, confidence, patient_info, csme_info=None):
    patient_messages = {
        0: "The automated screening model did not find signs of diabetic retinopathy in this image. Continue good blood sugar control and keep up regular eye screening (about every 12 months, or as your doctor advises).",
        1: "The automated screening model suggests mild diabetic retinopathy, the earliest stage. Vision is usually unaffected at this stage. Work with your doctor on blood sugar control and repeat screening in 6 to 12 months.",
        2: "The automated screening model suggests moderate diabetic retinopathy. Please arrange a dilated eye examination with an ophthalmologist within 2 to 4 months.",
        3: "The automated screening model suggests severe diabetic retinopathy. Please see a retina specialist within 2 to 4 weeks.",
        4: "The automated screening model suggests proliferative diabetic retinopathy, the most advanced stage. Please seek urgent evaluation by an eye specialist."
    }

    csme_str = csme_info.get("name", "CSME Low / Negative") if csme_info else "CSME Low / Negative"
    csme_action = csme_info.get("action", "") if csme_info else ""

    soap_note = f"""**Subjective:** {patient_info.get('age')} yo patient. Diabetes duration: {patient_info.get('duration')} years. HbA1c: {patient_info.get('hba1c')}%.
**Objective:** Fundus image passed automated quality heuristics. Screening output: {stage_name} ({confidence:.1f}% calibrated confidence). Macular Status: {csme_str}. The Grad-CAM++ map highlights microvascular activation regions.
**Assessment:** AI-suggested diabetic retinopathy: {stage_name} with {csme_str}. Requires clinician confirmation.
**Plan (suggested):** {DIAGNOSIS_INFO[stage_idx]['action']} {csme_action}"""

    return (
        f"### Patient Summary\n{patient_messages.get(stage_idx, '')}\n\n"
        f"### Clinical SOAP Note\n{soap_note}\n\n"
        f"*{DISCLAIMER_TEXT}*"
    )

def _gemini_notes(stage_name, confidence, patient_info, api_key, csme_info=None):
    # No patient name or other direct identifiers are sent to the external API.
    client = genai.Client(api_key=api_key)
    csme_str = csme_info.get("name", "CSME Low / Negative") if csme_info else "CSME Low / Negative"
    prompt = f"""
You are an assistant helping draft notes for a diabetic retinopathy screening report.
Patient: {patient_info.get('age')} years old, diabetes duration {patient_info.get('duration')} years, HbA1c {patient_info.get('hba1c')}%.
Automated screening result: {stage_name} with {confidence:.1f}% calibrated confidence.
Macular Edema (CSME) Status: {csme_str}.

Rules:
- Base your text ONLY on the screening result above.
- Explicitly report both Retinopathy Stage and Macular Edema (CSME) risk.
- State clearly that this is an automated screening result that must be confirmed by an ophthalmologist.
- Be concise and avoid unsupported medical claims.

Write two sections with these exact headings:
### Patient Explanation
(clear, empathetic, no jargon)
### Clinical SOAP Note
(Subjective, Objective, Assessment, Plan for the attending doctor)
"""
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text

def generate_clinical_notes(stage_idx, stage_name, confidence, patient_info, api_key=None, csme_info=None):
    """Returns (notes_text, used_gemini, error_message). Gemini results are cached per session
    so Streamlit reruns do not trigger repeated API calls."""
    error_msg = None
    csme_key_str = csme_info.get("key", "Low") if csme_info else "Low"
    if api_key and GENAI_AVAILABLE:
        cache = st.session_state.setdefault("gemini_cache", {})
        key = (stage_idx, round(float(confidence), 1), patient_info.get('age'),
               patient_info.get('duration'), patient_info.get('hba1c'), csme_key_str)
        if key in cache:
            return cache[key], True, None
        try:
            text = _gemini_notes(stage_name, confidence, patient_info, api_key, csme_info=csme_info)
            if text:
                cache[key] = text
                return text, True, None
            error_msg = "Gemini returned an empty response."
        except Exception as e:
            error_msg = f"Gemini request failed ({type(e).__name__}); showing the offline summary instead."
    elif api_key and not GENAI_AVAILABLE:
        error_msg = "google-genai is not installed; showing the offline summary instead."

    return _offline_notes(stage_idx, stage_name, confidence, patient_info, csme_info=csme_info), False, error_msg

def get_sample_image(stage_num, case_idx=0):
    """Finds and loads the image for stage_num and case_idx from sample_images directory."""
    if not os.path.exists("sample_images"):
        return None, ""
    matched = []
    for root, dirs, files in os.walk("sample_images"):
        for f in sorted(files):
            if f.lower().startswith(f"stage_{stage_num}") and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                matched.append(os.path.join(root, f))
    if 0 <= case_idx < len(matched):
        path = matched[case_idx]
        img = cv2.imread(path)
        return img, path
    return None, ""

def _offline_bilateral_notes(patient_info, od_stage, od_conf, os_stage, os_conf, od_csme=None, os_csme=None):
    overall_stage = max(od_stage, os_stage)
    diff = abs(od_stage - os_stage)
    
    higher_eye_str = "Right Eye (OD)" if od_stage > os_stage else "Left Eye (OS)" if os_stage > od_stage else "Both Eyes (OD & OS)"
    od_csme_str = od_csme.get("name", "CSME Low / Negative") if od_csme else "CSME Low / Negative"
    os_csme_str = os_csme.get("name", "CSME Low / Negative") if os_csme else "CSME Low / Negative"
    
    if diff == 0:
        asym_text = f"Bilateral concordant involvement: Both eyes present with {DIAGNOSIS_INFO[overall_stage]['name']}."
        plan_extra = "Routine bilateral monitoring per standard protocol."
    elif diff == 1:
        asym_text = f"Mild inter-ocular asymmetry: {higher_eye_str} is at Stage {overall_stage}, while the contralateral eye is at Stage {min(od_stage, os_stage)}. Overall triage urgency is governed by {higher_eye_str}."
        plan_extra = f"Prioritize specialist consultation for {higher_eye_str}; monitor contralateral eye closely."
    else:
        asym_text = f"Marked inter-ocular asymmetry (Δ = {diff} stages): {higher_eye_str} demonstrates Stage {overall_stage}, whereas the contralateral eye is at Stage {min(od_stage, os_stage)}."
        plan_extra = f"Urgent consultation for {higher_eye_str}. Recommend systemic vascular assessment (carotid artery Doppler ultrasonography) to evaluate for ocular ischemic syndrome or unilateral microvascular risk factors."

    patient_summary = (
        f"The screening examined both eyes (OD: Right Eye, OS: Left Eye). "
        f"Right Eye (OD) was assessed as {DIAGNOSIS_INFO[od_stage]['name']} ({od_conf:.1f}% confidence, {od_csme_str}), "
        f"and Left Eye (OS) was assessed as {DIAGNOSIS_INFO[os_stage]['name']} ({os_conf:.1f}% confidence, {os_csme_str}). "
        f"Overall clinical triage is determined by the more severe eye ({higher_eye_str}), "
        f"categorizing overall patient care as {DIAGNOSIS_INFO[overall_stage]['name']}."
    )

    soap_note = f"""**Subjective:** {patient_info.get('age')} yo patient. Diabetes duration: {patient_info.get('duration')} years. HbA1c: {patient_info.get('hba1c')}%.
**Objective:** Bilateral retinal screening with automated quality checks.
- Right Eye (OD): {DIAGNOSIS_INFO[od_stage]['name']} ({od_conf:.1f}% calibrated confidence) | Macula: {od_csme_str}.
- Left Eye (OS): {DIAGNOSIS_INFO[os_stage]['name']} ({os_conf:.1f}% calibrated confidence) | Macula: {os_csme_str}.
- Inter-Ocular Correlation: {asym_text}
Grad-CAM++ maps indicate microvascular attention regions; clinician verification required.
**Assessment:** AI-suggested patient diagnosis: {DIAGNOSIS_INFO[overall_stage]['name']} (governed by {higher_eye_str}). Inter-ocular stage discrepancy: Δ = {diff}.
**Plan (suggested):** {DIAGNOSIS_INFO[overall_stage]['action']} {plan_extra}"""

    return (
        f"### Bilateral Patient Summary\n{patient_summary}\n\n"
        f"### Clinical SOAP Note (Bilateral)\n{soap_note}\n\n"
        f"*{DISCLAIMER_TEXT}*"
    )

def _gemini_bilateral_notes(patient_info, od_stage, od_conf, os_stage, os_conf, api_key, od_csme=None, os_csme=None):
    client = genai.Client(api_key=api_key)
    overall_stage = max(od_stage, os_stage)
    diff = abs(od_stage - os_stage)
    higher_eye_str = "Right Eye (OD)" if od_stage > os_stage else "Left Eye (OS)" if os_stage > od_stage else "Both Eyes (OD & OS)"
    od_csme_str = od_csme.get("name", "CSME Low / Negative") if od_csme else "CSME Low / Negative"
    os_csme_str = os_csme.get("name", "CSME Low / Negative") if os_csme else "CSME Low / Negative"
    prompt = f"""
You are an ophthalmic clinical assistant drafting notes for a bilateral diabetic retinopathy screening report.
Patient: {patient_info.get('age')} years old, diabetes duration {patient_info.get('duration')} years, HbA1c {patient_info.get('hba1c')}%.
Screening protocol: Bilateral (OD vs. OS).
Right Eye (OD): {DIAGNOSIS_INFO[od_stage]['name']} with {od_conf:.1f}% calibrated confidence. Macular Status: {od_csme_str}.
Left Eye (OS): {DIAGNOSIS_INFO[os_stage]['name']} with {os_conf:.1f}% calibrated confidence. Macular Status: {os_csme_str}.
Overall Patient Stage: {DIAGNOSIS_INFO[overall_stage]['name']} (governed by {higher_eye_str}).
Inter-Ocular Discrepancy: Δ = {diff} stages.

Rules:
- Base text ONLY on the automated screening results above.
- Mention both OD and OS findings, including Retinopathy Stage and Macular Edema (CSME) risk.
- Explain that overall urgency is dictated by the worse eye.
- If inter-ocular discrepancy is >= 2 stages, note that marked asymmetry warrants systemic/carotid Doppler evaluation.
- State clearly that this is an automated screening result that requires clinician confirmation.
- Be concise and medically professional.

Write two sections with these exact headings:
### Bilateral Patient Explanation
(clear, empathetic, no jargon, explaining both eyes)
### Clinical SOAP Note (Bilateral)
(Subjective, Objective, Assessment, Plan for the attending ophthalmologist)
"""
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text

def generate_bilateral_clinical_notes(patient_info, od_stage, od_conf, os_stage, os_conf, api_key=None, od_csme=None, os_csme=None):
    error_msg = None
    od_c_key = od_csme.get("key", "Low") if od_csme else "Low"
    os_c_key = os_csme.get("key", "Low") if os_csme else "Low"
    if api_key and GENAI_AVAILABLE:
        cache = st.session_state.setdefault("gemini_cache", {})
        key = ("bilateral", od_stage, round(float(od_conf), 1), os_stage, round(float(os_conf), 1),
               patient_info.get('age'), patient_info.get('duration'), patient_info.get('hba1c'), od_c_key, os_c_key)
        if key in cache:
            return cache[key], True, None
        try:
            text = _gemini_bilateral_notes(patient_info, od_stage, od_conf, os_stage, os_conf, api_key, od_csme=od_csme, os_csme=os_csme)
            if text:
                cache[key] = text
                return text, True, None
            error_msg = "Gemini returned an empty response."
        except Exception as e:
            error_msg = f"Gemini request failed ({type(e).__name__}); showing the offline summary instead."
    elif api_key and not GENAI_AVAILABLE:
        error_msg = "google-genai is not installed; showing the offline summary instead."

    return _offline_bilateral_notes(patient_info, od_stage, od_conf, os_stage, os_conf, od_csme=od_csme, os_csme=os_csme), False, error_msg

# ----------------- PDF REPORT BUILDER -----------------
_UNSUPPORTED_CHARS = re.compile(r'[^\x00-\x7F\u00A0-\u00FF\u2013\u2014\u2018\u2019\u201C\u201D\u2022]')

def _inline_markup(line):
    """Convert one line of simple markdown to ReportLab markup (escaped, no emoji)."""
    line = _UNSUPPORTED_CHARS.sub('', line).strip()
    bullet = ''
    if re.match(r'^[\*\-]\s+', line):
        line = re.sub(r'^[\*\-]\s+', '', line)
        bullet = '&bull; '
    line = html.escape(line)
    line = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', line)
    line = re.sub(r'(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)', r'<i>\1</i>', line)
    line = line.replace('*', '')
    return bullet + line

def _notes_to_flowables(text, body_style, header_style):
    flowables = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            flowables.append(Spacer(1, 4))
        elif stripped.startswith('#'):
            heading = stripped.lstrip('#').strip().replace('**', '')
            flowables.append(Paragraph(f"<b>{_inline_markup(heading)}</b>", header_style))
        else:
            flowables.append(Paragraph(_inline_markup(stripped), body_style))
    return flowables

def _jpeg_buffer(rgb_array):
    buf = io.BytesIO()
    Image.fromarray(rgb_array).save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return buf

def generate_pdf_report(patient_info, stage_name, confidence, original_rgb, gradcam_rgb, clinical_notes_text, stage_idx=0, csme_info=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontSize=18, leading=22,
        textColor=colors.HexColor('#1E3A8A'), alignment=1, spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSub', parent=styles['Normal'], fontSize=9,
        textColor=colors.HexColor('#64748B'), alignment=1, spaceAfter=12
    )
    header_style = ParagraphStyle(
        'SectionHead', parent=styles['Heading2'], fontSize=11, leading=14,
        textColor=colors.HexColor('#1E3A8A'), spaceBefore=8, spaceAfter=4
    )
    body_style = ParagraphStyle(
        'DocBody', parent=styles['Normal'], fontSize=8.5, leading=11,
        textColor=colors.HexColor('#1E293B')
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer', parent=body_style, fontSize=7.5, leading=10,
        textColor=colors.HexColor('#B91C1C')
    )

    # 1. Header
    elements.append(Paragraph("<b>RETINA INSIGHT SCREENING REPORT</b>", title_style))
    elements.append(Paragraph("Automated Retinal Screening, Macular Edema (CSME) &amp; Explainable AI", subtitle_style))
    elements.append(Spacer(1, 8))

    # 2. Patient table
    scan_date = datetime.now().strftime("%d %b %Y, %I:%M %p")
    p_name_esc = html.escape(str(patient_info.get('name', 'Anonymous')))
    patient_data = [
        [Paragraph("<b>Patient Name:</b>", body_style), Paragraph(p_name_esc, body_style),
         Paragraph("<b>Report Date:</b>", body_style), Paragraph(scan_date, body_style)],
        [Paragraph("<b>Age / Gender:</b>", body_style), Paragraph(f"{patient_info.get('age', 'N/A')} yrs / {html.escape(str(patient_info.get('gender', 'N/A')))}", body_style),
         Paragraph("<b>HbA1c Level:</b>", body_style), Paragraph(f"{patient_info.get('hba1c', 'N/A')}%", body_style)],
        [Paragraph("<b>Diabetes Duration:</b>", body_style), Paragraph(f"{patient_info.get('duration', 'N/A')} Years", body_style),
         Paragraph("<b>Image Quality Check:</b>", body_style), Paragraph("Passed (automated heuristic)", body_style)],
        [Paragraph("<b>Location / Area:</b>", body_style), Paragraph(html.escape(str(patient_info.get('location', 'N/A'))), body_style),
         Paragraph("<b>Triage Pathway:</b>", body_style), Paragraph(html.escape(str(CLINICAL_PROTOCOLS.get(int(stage_idx), {}).get('timeline', 'Routine'))), body_style)]
    ]
    t_patient = Table(patient_data, colWidths=[100, 160, 100, 160])
    t_patient.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(t_patient)
    elements.append(Spacer(1, 10))

    # 3. Severity callout
    csme_text = html.escape(str(csme_info.get('name', 'CSME Low / Negative'))) if csme_info else "CSME Low / Negative"
    diag_data = [
        [Paragraph(f"<b>AI-SUGGESTED STAGE:</b> {html.escape(stage_name.upper())}<br/><font size='8' color='#93C5FD'><b>Macular Status:</b> {csme_text}</font>", ParagraphStyle('W', parent=body_style, fontSize=9.5, textColor=colors.white)),
         Paragraph(f"<b>Calibrated Confidence:</b> {confidence:.1f}%", ParagraphStyle('W2', parent=body_style, fontSize=9.5, textColor=colors.white, alignment=2))]
    ]
    t_diag = Table(diag_data, colWidths=[360, 160])
    t_diag.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#1E3A8A')),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(t_diag)
    elements.append(Spacer(1, 10))

    # 3b. Evidence-Based Clinical Referral Pathway (ICO/WHO Guidelines)
    proto = CLINICAL_PROTOCOLS.get(int(stage_idx), CLINICAL_PROTOCOLS[0])
    referral_rows = [
        [Paragraph("<b>Clinical Triage Urgency:</b>", body_style), Paragraph(f"<b>{html.escape(proto['urgency'])}</b>", body_style),
         Paragraph("<b>Recommended Timeline:</b>", body_style), Paragraph(f"<b>{html.escape(proto['timeline'])}</b>", body_style)],
        [Paragraph("<b>Referral Level:</b>", body_style), Paragraph(html.escape(proto['referral_level']), body_style),
         Paragraph("<b>Recommended Workup:</b>", body_style), Paragraph(html.escape(proto['workup']), body_style)]
    ]
    if csme_info:
        referral_rows.append([
            Paragraph("<b>Macular (DME) Risk:</b>", body_style), Paragraph(f"<b>{csme_text}</b>", body_style),
            Paragraph("<b>Macular Action:</b>", body_style), Paragraph(html.escape(str(csme_info.get('action', 'Routine macula evaluation'))), body_style)
        ])
    t_referral = Table(referral_rows, colWidths=[110, 150, 110, 150])
    t_referral.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FEF3C7') if int(stage_idx) >= 2 else colors.HexColor('#F0FDF4')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#F59E0B') if int(stage_idx) >= 2 else colors.HexColor('#86EFAC')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(t_referral)
    elements.append(Spacer(1, 10))

    # 4. Images (in-memory buffers: safe for multiple simultaneous users)
    img_data = [
        [Paragraph("<b>Original Retinal Scan</b>", body_style), Paragraph("<b>Grad-CAM Attention Map</b>", body_style)],
        [RLImage(_jpeg_buffer(original_rgb), width=200, height=200),
         RLImage(_jpeg_buffer(gradcam_rgb), width=200, height=200)]
    ]
    t_img = Table(img_data, colWidths=[260, 260])
    t_img.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 3),
    ]))
    elements.append(t_img)
    elements.append(Spacer(1, 8))

    # 5. Notes (markdown converted to ReportLab markup)
    elements.append(Paragraph("<b>Clinical Evaluation &amp; Decision Support</b>", header_style))
    elements.extend(_notes_to_flowables(clinical_notes_text, body_style, header_style))
    elements.append(Spacer(1, 12))

    # 6. Sign-off and disclaimer
    sign_data = [
        [Paragraph("<b>Screening Software:</b> Retina Insight v2.5", body_style),
         Paragraph("<b>Reviewing Ophthalmologist Sign-off:</b> ___________________", body_style)]
    ]
    t_sign = Table(sign_data, colWidths=[280, 240])
    elements.append(t_sign)
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(f"<b>Disclaimer:</b> {html.escape(DISCLAIMER_TEXT)}", disclaimer_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer

def generate_bilateral_pdf_report(
    patient_info,
    od_stage_idx, od_confidence, od_orig_rgb, od_cam_rgb, od_blur,
    os_stage_idx, os_confidence, os_orig_rgb, os_cam_rgb, os_blur,
    clinical_notes_text,
    od_csme=None, os_csme=None
):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontSize=16, leading=20,
        textColor=colors.HexColor('#1E3A8A'), alignment=1, spaceAfter=3
    )
    subtitle_style = ParagraphStyle(
        'DocSub', parent=styles['Normal'], fontSize=8.5,
        textColor=colors.HexColor('#64748B'), alignment=1, spaceAfter=8
    )
    header_style = ParagraphStyle(
        'SectionHead', parent=styles['Heading2'], fontSize=10.5, leading=13,
        textColor=colors.HexColor('#1E3A8A'), spaceBefore=6, spaceAfter=3
    )
    body_style = ParagraphStyle(
        'DocBody', parent=styles['Normal'], fontSize=8, leading=10.5,
        textColor=colors.HexColor('#1E293B')
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer', parent=body_style, fontSize=7, leading=9.5,
        textColor=colors.HexColor('#B91C1C')
    )

    overall_stage = max(od_stage_idx, os_stage_idx)
    diff = abs(od_stage_idx - os_stage_idx)

    # 1. Header
    elements.append(Paragraph("<b>RETINA INSIGHT CLINICAL REPORT &mdash; BILATERAL SCREENING</b>", title_style))
    elements.append(Paragraph("Comprehensive Multi-Eye Workup (OD vs. OS), Inter-Ocular Asymmetry &amp; Macular Status", subtitle_style))
    elements.append(Spacer(1, 4))

    # 2. Patient Table
    scan_date = datetime.now().strftime("%d %b %Y, %I:%M %p")
    p_name_esc = html.escape(str(patient_info.get('name', 'Anonymous')))
    patient_data = [
        [Paragraph("<b>Patient Name:</b>", body_style), Paragraph(p_name_esc, body_style),
         Paragraph("<b>Report Date:</b>", body_style), Paragraph(scan_date, body_style)],
        [Paragraph("<b>Age / Gender:</b>", body_style), Paragraph(f"{patient_info.get('age', 'N/A')} yrs / {html.escape(str(patient_info.get('gender', 'N/A')))}", body_style),
         Paragraph("<b>HbA1c Level:</b>", body_style), Paragraph(f"{patient_info.get('hba1c', 'N/A')}%", body_style)],
        [Paragraph("<b>Diabetes Duration:</b>", body_style), Paragraph(f"{patient_info.get('duration', 'N/A')} Years", body_style),
         Paragraph("<b>Screening Protocol:</b>", body_style), Paragraph("Bilateral (OD + OS Workup)", body_style)],
        [Paragraph("<b>Location / Area:</b>", body_style), Paragraph(html.escape(str(patient_info.get('location', 'N/A'))), body_style),
         Paragraph("<b>Overall Triage Urgency:</b>", body_style), Paragraph(html.escape(str(CLINICAL_PROTOCOLS[overall_stage]['urgency'])), body_style)]
    ]
    t_patient = Table(patient_data, colWidths=[95, 175, 95, 175])
    t_patient.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(t_patient)
    elements.append(Spacer(1, 6))

    # 3. Overall Patient Diagnosis Banner
    governing_eye = "Right Eye (OD)" if od_stage_idx > os_stage_idx else "Left Eye (OS)" if os_stage_idx > od_stage_idx else "Bilateral Concordant (OD & OS)"
    diag_banner = [
        [Paragraph(f"<b>OVERALL PATIENT DIAGNOSIS:</b> {html.escape(DIAGNOSIS_INFO[overall_stage]['name'].upper())}",
                   ParagraphStyle('W1', parent=body_style, fontSize=9.5, textColor=colors.white)),
         Paragraph(f"<b>Governing Eye:</b> {governing_eye}",
                   ParagraphStyle('W2', parent=body_style, fontSize=8.5, textColor=colors.white, alignment=2))]
    ]
    t_diag = Table(diag_banner, colWidths=[360, 180])
    t_diag.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#1E3A8A')),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(t_diag)
    elements.append(Spacer(1, 6))

    # 4. Bilateral Comparative Findings Table
    asym_status = "Concordant (Symmetric)" if diff == 0 else f"Mild Asymmetry (Δ={diff})" if diff == 1 else f"Marked Asymmetry (Δ={diff} stages)"
    od_csme_label = od_csme.get('name', 'CSME Low') if od_csme else 'CSME Low'
    os_csme_label = os_csme.get('name', 'CSME Low') if os_csme else 'CSME Low'

    comp_data = [
        [Paragraph("<b>Diagnostic Parameter</b>", body_style),
         Paragraph("<b>Right Eye (OD - Oculus Dexter)</b>", body_style),
         Paragraph("<b>Left Eye (OS - Oculus Sinister)</b>", body_style),
         Paragraph("<b>Bilateral Synthesis</b>", body_style)],
        [Paragraph("<b>Predicted DR Severity:</b>", body_style),
         Paragraph(f"<b>{html.escape(DIAGNOSIS_INFO[od_stage_idx]['name'])}</b>", body_style),
         Paragraph(f"<b>{html.escape(DIAGNOSIS_INFO[os_stage_idx]['name'])}</b>", body_style),
         Paragraph(f"Governed by Stage {overall_stage}", body_style)],
        [Paragraph("<b>CSME Macular Risk:</b>", body_style),
         Paragraph(html.escape(od_csme_label), body_style),
         Paragraph(html.escape(os_csme_label), body_style),
         Paragraph("OCT Macula Workup", body_style)],
        [Paragraph("<b>Calibrated Confidence:</b>", body_style),
         Paragraph(f"{od_confidence:.1f}%", body_style),
         Paragraph(f"{os_confidence:.1f}%", body_style),
         Paragraph(asym_status, body_style)],
        [Paragraph("<b>Acquisition Quality:</b>", body_style),
         Paragraph(f"Passed (Blur: {od_blur:.1f})", body_style),
         Paragraph(f"Passed (Blur: {os_blur:.1f})", body_style),
         Paragraph("Both Eyes Analyzable", body_style)],
        [Paragraph("<b>Eye-Specific Urgency:</b>", body_style),
         Paragraph(html.escape(CLINICAL_PROTOCOLS[od_stage_idx]['urgency']), body_style),
         Paragraph(html.escape(CLINICAL_PROTOCOLS[os_stage_idx]['urgency']), body_style),
         Paragraph(f"<b>{html.escape(CLINICAL_PROTOCOLS[overall_stage]['timeline'])}</b>", body_style)]
    ]
    t_comp = Table(comp_data, colWidths=[120, 145, 145, 130])
    t_comp.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FFFFFF')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#94A3B8')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(t_comp)
    elements.append(Spacer(1, 6))

    # 4b. Asymmetry Red-Flag Box if diff >= 2
    if diff >= 2:
        asym_alert = [
            [Paragraph(f"<b>CLINICAL RED FLAG &mdash; MARKED INTER-OCULAR ASYMMETRY (Δ = {diff} STAGES):</b><br/>"
                       f"The patient demonstrates significant disparity between the right and left eyes. Highly asymmetric "
                       f"diabetic retinopathy requires clinical investigation for ipsilateral carotid artery occlusive disease "
                       f"(Ocular Ischemic Syndrome), unilateral glaucoma, or previous laser intervention.",
                       ParagraphStyle('RedAlert', parent=body_style, fontSize=7.5, textColor=colors.HexColor('#991B1B')))]
        ]
        t_asym = Table(asym_alert, colWidths=[540])
        t_asym.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FEE2E2')),
            ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#EF4444')),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(t_asym)
        elements.append(Spacer(1, 6))

    # 5. Evidence-Based Clinical Referral Pathway (ICO/WHO Guidelines)
    proto = CLINICAL_PROTOCOLS.get(int(overall_stage), CLINICAL_PROTOCOLS[0])
    referral_rows = [
        [Paragraph("<b>Clinical Triage Urgency:</b>", body_style), Paragraph(f"<b>{html.escape(proto['urgency'])}</b>", body_style),
         Paragraph("<b>Recommended Timeline:</b>", body_style), Paragraph(f"<b>{html.escape(proto['timeline'])}</b>", body_style)],
        [Paragraph("<b>Referral Care Level:</b>", body_style), Paragraph(html.escape(proto['referral_level']), body_style),
         Paragraph("<b>Recommended Workup:</b>", body_style), Paragraph(html.escape(proto['workup']), body_style)]
    ]
    t_referral = Table(referral_rows, colWidths=[110, 160, 110, 160])
    t_referral.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FEF3C7') if int(overall_stage) >= 2 else colors.HexColor('#F0FDF4')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#F59E0B') if int(overall_stage) >= 2 else colors.HexColor('#86EFAC')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(t_referral)
    elements.append(Spacer(1, 6))

    # 6. Images: 4 columns side by side
    img_data = [
        [Paragraph("<b>OD: Original Fundus</b>", body_style),
         Paragraph("<b>OD: Grad-CAM++ Attention</b>", body_style),
         Paragraph("<b>OS: Original Fundus</b>", body_style),
         Paragraph("<b>OS: Grad-CAM++ Attention</b>", body_style)],
        [RLImage(_jpeg_buffer(od_orig_rgb), width=128, height=128),
         RLImage(_jpeg_buffer(od_cam_rgb), width=128, height=128),
         RLImage(_jpeg_buffer(os_orig_rgb), width=128, height=128),
         RLImage(_jpeg_buffer(os_cam_rgb), width=128, height=128)]
    ]
    t_img = Table(img_data, colWidths=[135, 135, 135, 135])
    t_img.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 1),
    ]))
    elements.append(t_img)
    elements.append(Spacer(1, 6))

    # 7. Bilateral Clinical SOAP Notes
    elements.append(Paragraph("<b>Clinical Evaluation &amp; Bilateral Decision Support Notes</b>", header_style))
    elements.extend(_notes_to_flowables(clinical_notes_text, body_style, header_style))
    elements.append(Spacer(1, 8))

    # 8. Sign-off and disclaimer
    sign_data = [
        [Paragraph("<b>Screening Software:</b> Retina Insight v2.5 (Bilateral Screening Module)", body_style),
         Paragraph("<b>Reviewing Ophthalmologist Sign-off:</b> ___________________", body_style)]
    ]
    t_sign = Table(sign_data, colWidths=[290, 250])
    elements.append(t_sign)
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(f"<b>Disclaimer:</b> {html.escape(DISCLAIMER_TEXT)}", disclaimer_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ----------------- SIDEBAR -----------------
st.sidebar.header("👁️ Screening Protocol & Input")
screening_mode = st.sidebar.radio(
    "Select Screening Protocol:",
    [
        "Clinical Bilateral Screening (OD & OS)",
        "Single Fundus Scan",
        "Screening Camp Batch Processing (Bulk Triage)"
    ],
    index=0,
    help="Select between standard bilateral clinical workup, single-eye inspection, or high-throughput screening camp batch processing."
)

img_bgr = None
active_source_label = ""
img_bgr_od = None
od_source_label = ""
img_bgr_os = None
os_source_label = ""

camp_name = "Community Vision Screening Camp #4"
screener_id = "Dr. A. Rao / Lead Screener"
camp_loc = "Hyderabad Rural / Medchal"
batch_scans_data = [] # list of (display_name, img_bgr)

if screening_mode == "Single Fundus Scan":
    st.sidebar.subheader("📁 Fundus Input (Single Eye)")
    input_mode = st.sidebar.radio("Input Source:", ["Clinical Database", "Upload Custom Scan"])

    if input_mode == "Upload Custom Scan":
        uploaded_file = st.sidebar.file_uploader("Upload fundus photograph (.png, .jpg)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img_bgr is None:
                st.sidebar.error("❌ Could not read this file as an image. Please upload a valid PNG or JPG.")
            else:
                active_source_label = f"Uploaded File: {uploaded_file.name}"
    else:
        sel_stage = st.sidebar.selectbox(
            "Select Severity Stage:",
            ["Stage 0 (No DR)", "Stage 1 (Mild DR)", "Stage 2 (Moderate DR)", "Stage 3 (Severe DR)", "Stage 4 (Proliferative DR)"]
        )
        stage_num = int(sel_stage.split()[1])

        matched_images = []
        if os.path.exists("sample_images"):
            for root, dirs, files in os.walk("sample_images"):
                for f in sorted(files):
                    if f.lower().startswith(f"stage_{stage_num}") and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        matched_images.append(os.path.join(root, f))

        if len(matched_images) > 0:
            case_options = [f"Patient Case {i+1}" for i in range(len(matched_images))]
            sel_case = st.sidebar.selectbox("Select Patient Case:", case_options)
            case_idx = int(sel_case.split()[-1]) - 1

            chosen_path = matched_images[case_idx]
            img_bgr = cv2.imread(chosen_path)
            if img_bgr is None:
                st.sidebar.error(f"❌ Could not read {chosen_path}.")
            else:
                active_source_label = f"Database: {sel_stage} — {sel_case}"
                st.sidebar.caption(f"📁 {len(matched_images)} case(s) available for Stage {stage_num}")
        else:
            st.sidebar.error(f"❌ No images found for Stage {stage_num} in sample_images folder.")
elif screening_mode == "Clinical Bilateral Screening (OD & OS)":
    st.sidebar.subheader("📁 Bilateral Fundus Input (OD vs. OS)")
    bilateral_source = st.sidebar.radio(
        "Bilateral Input Source:",
        ["Clinical Scenarios & Presets", "Upload Dual Scans (OD & OS)"]
    )
    if bilateral_source == "Clinical Scenarios & Presets":
        preset_choice = st.sidebar.selectbox(
            "Select Clinical Scenario / Preset:",
            [
                "Preset 1: Bilateral Moderate DR (OD Stage 2, OS Stage 2)",
                "Preset 2: Asymmetric Progression (OD Stage 3 Severe, OS Stage 1 Mild)",
                "Preset 3: High-Risk Proliferative Asymmetry (OD Stage 4 PDR, OS Stage 2 Moderate)",
                "Preset 4: Early Unilateral Retinopathy (OD Stage 1 Mild, OS Stage 0 Normal)",
                "Preset 5: Bilateral Healthy Retina (OD Stage 0 Normal, OS Stage 0 Normal)",
                "Custom Bilateral Selection (OD & OS Pickers)"
            ]
        )
        if preset_choice == "Preset 1: Bilateral Moderate DR (OD Stage 2, OS Stage 2)":
            img_bgr_od, _ = get_sample_image(2, case_idx=0)
            img_bgr_os, _ = get_sample_image(2, case_idx=1)
            od_source_label = "OD: Stage 2 Case 1"
            os_source_label = "OS: Stage 2 Case 2"
        elif preset_choice == "Preset 2: Asymmetric Progression (OD Stage 3 Severe, OS Stage 1 Mild)":
            img_bgr_od, _ = get_sample_image(3, case_idx=0)
            img_bgr_os, _ = get_sample_image(1, case_idx=0)
            od_source_label = "OD: Stage 3 Case 1"
            os_source_label = "OS: Stage 1 Case 1"
        elif preset_choice == "Preset 3: High-Risk Proliferative Asymmetry (OD Stage 4 PDR, OS Stage 2 Moderate)":
            img_bgr_od, _ = get_sample_image(4, case_idx=0)
            img_bgr_os, _ = get_sample_image(2, case_idx=0)
            od_source_label = "OD: Stage 4 Case 1"
            os_source_label = "OS: Stage 2 Case 1"
        elif preset_choice == "Preset 4: Early Unilateral Retinopathy (OD Stage 1 Mild, OS Stage 0 Normal)":
            img_bgr_od, _ = get_sample_image(1, case_idx=0)
            img_bgr_os, _ = get_sample_image(0, case_idx=0)
            od_source_label = "OD: Stage 1 Case 1"
            os_source_label = "OS: Stage 0 Case 1"
        elif preset_choice == "Preset 5: Bilateral Healthy Retina (OD Stage 0 Normal, OS Stage 0 Normal)":
            img_bgr_od, _ = get_sample_image(0, case_idx=0)
            img_bgr_os, _ = get_sample_image(0, case_idx=1)
            od_source_label = "OD: Stage 0 Case 1"
            os_source_label = "OS: Stage 0 Case 2"
        else: # Custom Bilateral Selection
            st.sidebar.markdown("**Right Eye (OD — Oculus Dexter):**")
            od_st = st.sidebar.selectbox("OD Stage:", ["Stage 0", "Stage 1", "Stage 2", "Stage 3", "Stage 4"], index=2, key="od_st_sel")
            od_stage_num = int(od_st.split()[1])
            od_cs = st.sidebar.selectbox("OD Patient Case:", [f"Case {i+1}" for i in range(5)], index=0, key="od_cs_sel")
            od_case_idx = int(od_cs.split()[-1]) - 1
            img_bgr_od, _ = get_sample_image(od_stage_num, od_case_idx)
            od_source_label = f"OD: {od_st} {od_cs}"

            st.sidebar.markdown("**Left Eye (OS — Oculus Sinister):**")
            os_st = st.sidebar.selectbox("OS Stage:", ["Stage 0", "Stage 1", "Stage 2", "Stage 3", "Stage 4"], index=1, key="os_st_sel")
            os_stage_num = int(os_st.split()[1])
            os_cs = st.sidebar.selectbox("OS Patient Case:", [f"Case {i+1}" for i in range(5)], index=0, key="os_cs_sel")
            os_case_idx = int(os_cs.split()[-1]) - 1
            img_bgr_os, _ = get_sample_image(os_stage_num, os_case_idx)
            os_source_label = f"OS: {os_st} {os_cs}"
    else: # Upload Dual Scans
        up_od = st.sidebar.file_uploader("Upload Right Eye (OD) Fundus Scan", type=["png", "jpg", "jpeg"], key="up_od_file")
        if up_od is not None:
            od_bytes = np.asarray(bytearray(up_od.read()), dtype=np.uint8)
            img_bgr_od = cv2.imdecode(od_bytes, cv2.IMREAD_COLOR)
            od_source_label = f"Uploaded OD: {up_od.name}"

        up_os = st.sidebar.file_uploader("Upload Left Eye (OS) Fundus Scan", type=["png", "jpg", "jpeg"], key="up_os_file")
        if up_os is not None:
            os_bytes = np.asarray(bytearray(up_os.read()), dtype=np.uint8)
            img_bgr_os = cv2.imdecode(os_bytes, cv2.IMREAD_COLOR)
            os_source_label = f"Uploaded OS: {up_os.name}"
else:
    # ----------------- SCREENING CAMP BATCH CONFIG -----------------
    st.sidebar.subheader("🏥 Screening Camp Configuration")
    camp_name = st.sidebar.text_input("Camp / Facility Name:", value="Community Vision Screening Camp #4")
    screener_id = st.sidebar.text_input("Screening Officer / ID:", value="Dr. A. Rao / Lead Screener")
    camp_loc = st.sidebar.text_input("Camp District / Location:", value="Hyderabad Rural / Medchal")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Batch Ingestion Method")
    camp_source = st.sidebar.radio(
        "Ingestion Source:",
        ["Load Clinical Camp Sample Batch (15 Cases)", "Upload Bulk Scans (Multi-Image Dropzone)"]
    )
    
    if camp_source == "Load Clinical Camp Sample Batch (15 Cases)":
        st.sidebar.caption("📁 Loads 15 verified clinical camp cases spanning Stages 0-4 from database.")
        if os.path.exists("sample_images"):
            # Load 3 cases from each of the 5 stages
            for stg in range(5):
                for cs in range(3):
                    img, path = get_sample_image(stg, cs)
                    if img is not None:
                        fname = os.path.basename(path)
                        batch_scans_data.append((f"PT-{stg*3 + cs + 1:03d}_{fname}", img))
    else:
        uploaded_batch = st.sidebar.file_uploader(
            "Upload Batch Retinal Scans (.png, .jpg)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True
        )
        if uploaded_batch:
            for up in uploaded_batch:
                f_bytes = np.asarray(bytearray(up.read()), dtype=np.uint8)
                im = cv2.imdecode(f_bytes, cv2.IMREAD_COLOR)
                if im is not None:
                    batch_scans_data.append((up.name, im))
            st.sidebar.success(f"✅ Loaded {len(batch_scans_data)} scan(s) from upload.")

# Patient Clinical Context (Single & Bilateral Modes)
if screening_mode != "Screening Camp Batch Processing (Bulk Triage)":
    st.sidebar.markdown("---")
    with st.sidebar.expander("👤 Patient Clinical Context", expanded=True):
        p_name = st.text_input("Patient Name", value="Ramesh Kumar")
        p_age = st.number_input("Age", min_value=18, max_value=100, value=54)
        p_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        p_duration = st.number_input("Diabetes Duration (Years)", min_value=0, max_value=50, value=9)
        p_hba1c = st.number_input("Recent HbA1c (%)", min_value=4.0, max_value=16.0, value=8.4, step=0.1)

    # Gemini API Key
    with st.sidebar.expander("🤖 Gemini GenAI Key (Optional)", expanded=False):
        gemini_key = st.text_input(
            "Gemini API Key", type="password",
            help="Leave empty to use the built-in offline summary. Patient name is never sent to Gemini."
        )
else:
    p_name = "Camp Screening Patient"
    p_age = 50
    p_gender = "Other"
    p_duration = 8
    p_hba1c = 8.0
    gemini_key = ""

# Inference & Quality Settings
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Architecture & Inference Controls")
backbone_choice = st.sidebar.selectbox(
    "Model Architecture Backbone:",
    [
        "EfficientNet-B0 (Calibrated Baseline - 100% Benchmark)",
        "Multi-Scale Clinical Ensemble (Standard + 4-Crop High-Precision TTA)"
    ],
    index=0,
    help="Switch between standard calibrated EfficientNet-B0 and multi-resolution clinical ensemble for difficult boundary cases."
)
use_multi_scale = ("Multi-Scale" in backbone_choice)

use_tta = st.sidebar.checkbox(
    "High-Precision TTA (Test-Time Augmentation)",
    value=True,
    help="Averages predictions across 4 rotational/flip invariant views to prevent boundary misclassifications (e.g. Stage 1 vs 2, Stage 2 vs 3)."
)
blur_threshold = st.sidebar.slider(
    "Blur Sensitivity Threshold",
    min_value=1.0,
    max_value=30.0,
    value=5.0,
    step=1.0,
    help="Images with standardized blur score below this threshold are rejected."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 System Benchmarks")
st.sidebar.markdown("**APTOS validation split**")
st.sidebar.markdown(f"- **QWK Kappa:** `{BENCHMARKS['qwk']}`")
st.sidebar.markdown(f"- **Accuracy:** `{BENCHMARKS['accuracy']}`")
st.sidebar.markdown(f"- **Calibration ECE:** `{BENCHMARKS['ece']}`")
st.sidebar.markdown("**External test (IDRiD, 103 images)**")
st.sidebar.markdown(f"- **Accuracy:** `{EXTERNAL['accuracy']}`")
st.sidebar.markdown(f"- **QWK Kappa:** `{EXTERNAL['qwk']}`")
st.sidebar.markdown(f"- **Referable-DR AUC:** `{EXTERNAL['auc_referable']}`")
st.sidebar.caption("Performance drops on images from other cameras and clinics.")
st.sidebar.markdown(f"- **Temperature ($T$):** `{temperature:.4f}`")
if not temp_from_file:
    st.sidebar.warning("temperature.txt missing or invalid: using the default temperature.")

# ----------------- MAIN SCREEN -----------------
if screening_mode == "Single Fundus Scan":
    if img_bgr is not None:
        st.info(f"🔍 **Active Image:** `{active_source_label}`")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        st.subheader("1. Image Quality Check (Automated Heuristics)")
        is_valid, msg, blur_val, bright_val = assess_image_quality(img_bgr, blur_threshold=blur_threshold)

        col1, col2, col3 = st.columns(3)
        with col1:
            if is_valid:
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")
        with col2:
            st.metric("Blur Score (Normalized)", f"{blur_val:.1f}")
        with col3:
            st.metric("Illumination Intensity", f"{bright_val:.1f}")

        if not is_valid:
            st.error(f"🚨 **Screening Halted:** {msg}")
            st.image(img_rgb, caption="Rejected Image", width=350)
        else:
            proc_rgb, calibrated_probs, pred_class, heatmap = run_inference(img_bgr, temperature, use_tta=use_tta, multi_scale=use_multi_scale)
            confidence = float(calibrated_probs[pred_class] * 100)
            csme_res = detect_macular_edema_risk(img_bgr, heatmap=heatmap, dr_stage=pred_class)

            st.markdown("---")
            st.subheader("2. Diagnostic Screening & Reliability")

            res1, res2 = st.columns([1, 1.5])
            with res1:
                st.write(f"### {DIAGNOSIS_INFO[pred_class]['name']}")
                st.progress(max(0, min(int(confidence) if not np.isnan(confidence) else 0, 100)))
                st.write(f"**Calibrated Confidence:** `{confidence:.2f}%`")
                if confidence < LOW_CONFIDENCE_THRESHOLD:
                    st.warning("⚠️ Low model confidence. Manual review by a clinician is strongly advised.")
                st.info(f"**Suggested action:** {DIAGNOSIS_INFO[pred_class]['action']}")

            with res2:
                chart_data = {f"Stage {i}": round(float(calibrated_probs[i] * 100), 2) for i in range(5)}
                st.bar_chart(chart_data)

            # CSME Macular Edema Risk Assessment Card
            csme_color = csme_res.get("badge_color", "green")
            border_hex = "#EF4444" if csme_color == "red" else "#F59E0B" if csme_color == "orange" else "#10B981"
            bg_hex = "#FEE2E2" if csme_color == "red" else "#FEF3C7" if csme_color == "orange" else "#ECFDF5"
            text_hex = "#991B1B" if csme_color == "red" else "#92400E" if csme_color == "orange" else "#065F46"

            st.markdown(
                f"""<div style="background-color: {bg_hex}; border: 1.5px solid {border_hex}; border-radius: 8px; padding: 12px 16px; margin: 12px 0;">
                <h4 style="margin: 0 0 6px 0; color: {text_hex};">🎯 Macular Assessment: {csme_res['status']}</h4>
                <p style="margin: 0; color: #1E293B; font-size: 13.5px;">
                <b>CSME Risk Score:</b> <code>{csme_res['score']:.1f}/100</code> &nbsp;|&nbsp; 
                <b>Central Foveal Exudates (&lt;500 μm):</b> <code>{csme_res['foveal_exudates']} px</code> &nbsp;|&nbsp; 
                <b>Macular Ring Exudates (1-2 DD):</b> <code>{csme_res['macular_exudates']} px</code>
                </p>
                <p style="margin: 6px 0 0 0; color: {text_hex}; font-size: 13px;"><b>Clinical Directive:</b> {csme_res['action']}</p>
                </div>""",
                unsafe_allow_html=True
            )

            st.markdown("---")
            st.subheader("3. Explainable AI (XAI) — Grad-CAM++ Attention & Micro-Lesion Localization")
            st.caption("Grad-CAM++ calculates positive second- and third-order partial derivatives to isolate scattered microaneurysms, hemorrhages, and exudates.")

            # Interactive Visualizer Controls
            ctrl1, ctrl2, ctrl3 = st.columns([1.5, 1.2, 1.3])
            with ctrl1:
                selected_cmap = st.selectbox(
                    "🎨 Clinical Heatmap Colormap:",
                    list(COLORMAP_OPTIONS.keys()),
                    index=0,
                    help="Switch between classic ophthalmic JET, high-contrast Inferno, Turbo, or colorblind-accessible Viridis."
                )
            with ctrl2:
                cam_alpha = st.slider(
                    "🔎 Heatmap Opacity (α):",
                    min_value=0.10,
                    max_value=0.90,
                    value=0.45,
                    step=0.05,
                    help="Adjust blend ratio between retinal fundus anatomy and lesion activation map."
                )
            with ctrl3:
                st.write("")
                st.write("")
                show_contours = st.checkbox(
                    "✨ Outline High-Attention Lesions",
                    value=False,
                    help="Draws white boundary contours around high-confidence activation zones (> 60% peak intensity)."
                )

            overlay = generate_cam_overlay(
                base_rgb=proc_rgb,
                heatmap=heatmap,
                colormap_name=selected_cmap,
                alpha=cam_alpha,
                show_contours=show_contours
            )

            short_cmap = selected_cmap.split()[0]
            contour_flag = " + Contours" if show_contours else ""

            t1, t2, t3, t4 = st.tabs([
                "1. Original Scan",
                "2. Ben Graham Preprocessed",
                f"3. Grad-CAM++ Attention ({short_cmap})",
                "4. Macular ROI & CSME Exudates"
            ])
            with t1:
                st.image(img_rgb, caption="1. Original Retinal Scan", use_container_width=True)
            with t2:
                st.image(proc_rgb, caption="2. Ben Graham Preprocessed (256x256)", use_container_width=True)
            with t3:
                st.image(overlay, caption=f"3. Grad-CAM++ ({short_cmap} | α={cam_alpha:.2f}{contour_flag})", use_container_width=True)
            with t4:
                st.image(csme_res["annotated_rgb"], caption="4. Macular Field: Foveal Zone (Cyan 500μm) + Outer Macula (Blue 1.5DD) + Detected Hard Exudates (Yellow)", use_container_width=True)

            # ----------------- SECTION 4: CLINICAL CARE PATHWAY & SPECIALIST LOCATOR -----------------
            st.markdown("---")
            st.subheader("4. Recommended Clinical Care Pathway & Specialist Referral")
            proto = CLINICAL_PROTOCOLS[pred_class]

            c_triage, c_timeline = st.columns([1.2, 1])
            with c_triage:
                if pred_class >= 2:
                    st.error(f"🚨 **Triage Priority:** {proto['urgency']}")
                elif pred_class == 1:
                    st.warning(f"⚠️ **Triage Priority:** {proto['urgency']}")
                else:
                    st.success(f"🟢 **Triage Priority:** {proto['urgency']}")
                st.write(f"**Referral Care Level:** `{proto['referral_level']}`")
                st.write(f"**Diagnostic Workup:** {proto['workup']}")

            with c_timeline:
                st.info(f"⏱️ **Follow-up Timeline:** {proto['timeline']}")
                if p_hba1c >= 8.0:
                    st.warning(f"⚠️ Patient HbA1c is **{p_hba1c:.1f}%** (elevated). Tight glycemic control with an endocrinologist is critical to prevent DR acceleration.")

            # Evidence-Based Systemic & Lifestyle Guidance
            with st.expander("📋 Evidence-Based Clinical Management Guidelines (ICO/WHO Standards)", expanded=True):
                st.write(f"**Clinical Summary:** {proto['summary']}")
                st.write("**Key Management Goals:**")
                for goal in proto['target_goals']:
                    st.markdown(f"- {goal}")
                st.caption("ℹ️ Guidance aligned with International Council of Ophthalmology (ICO) Diabetic Eye Care Guidelines.")

            # Nearby Eye Hospital & Retina Specialist Locator
            loc_clean = "Hyderabad"
            with st.expander("📍 Find Nearby Eye Hospitals & Retina Specialists (Primary Care Triage)", expanded=(pred_class >= 2)):
                st.markdown("For patients identified with referable or sight-threatening diabetic retinopathy, prompt referral to an accredited vitreo-retinal facility is essential.")
                
                loc_col1, loc_col2 = st.columns([2.5, 1])
                with loc_col1:
                    user_loc = st.text_input(
                        "Enter Patient's City, Area, or PIN Code:",
                        value="Hyderabad",
                        help="Enter any town, area, or city (e.g., Medchal, Secunderabad, Warangal, Karimnagar, Bangalore)."
                    )
                with loc_col2:
                    st.write("")
                    st.write("")
                    loc_clean = user_loc.strip() if user_loc and user_loc.strip() else "near me"

                encoded_loc = urllib.parse.quote_plus(f"retina specialist eye hospital in {loc_clean}")
                maps_url = f"https://www.google.com/maps/search/{encoded_loc}"
                
                st.link_button(
                    f"🔎 Search Eye Hospitals & Retina Specialists in '{loc_clean}' (Google Maps)",
                    url=maps_url,
                    use_container_width=True
                )
                st.markdown("#### Accredited Tertiary Eye Care Centers (Directory):")
                for region, hospitals in EYE_HOSPITALS_DIRECTORY.items():
                    st.markdown(f"**{region}:**")
                    for h in hospitals:
                        st.markdown(f"- **[{h['name']}]({h['url']})** — *{h['city']}* ({h['type']}) | 📞 `{h['contact']}`")

            # ----------------- SECTION 5: NOTES & PDF EXPORT -----------------
            st.markdown("---")
            st.subheader("5. Clinical Notes & PDF Report")

            patient_info_dict = {
                "name": p_name,
                "age": p_age,
                "gender": p_gender,
                "duration": p_duration,
                "hba1c": p_hba1c,
                "location": loc_clean
            }

            notes_text, is_gemini, notes_error = generate_clinical_notes(
                stage_idx=pred_class,
                stage_name=DIAGNOSIS_INFO[pred_class]['name'],
                confidence=confidence,
                patient_info=patient_info_dict,
                api_key=gemini_key.strip() if gemini_key else None,
                csme_info=csme_res
            )

            if notes_error:
                st.warning(f"ℹ️ {notes_error}")
            if is_gemini:
                st.success("🤖 Notes drafted by **Gemini (gemini-2.5-flash)** from the screening result only. Review before use.")
            else:
                st.caption("ℹ️ Notes generated by the built-in offline summary engine.")

            st.markdown(notes_text)

            st.markdown("#### 📄 Export Screening Report")
            try:
                pdf_bytes = generate_pdf_report(
                    patient_info=patient_info_dict,
                    stage_name=DIAGNOSIS_INFO[pred_class]['name'],
                    confidence=confidence,
                    original_rgb=cv2.resize(img_rgb, (300, 300)),
                    gradcam_rgb=cv2.resize(overlay, (300, 300)),
                    clinical_notes_text=notes_text,
                    stage_idx=pred_class,
                    csme_info=csme_res
                )

                safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', str(p_name).strip()) or "Patient"
                clean_filename = f"Retina_Report_{safe_name}_{datetime.now().strftime('%Y%m%d')}.pdf"
                st.download_button(
                    label="📥 Download Screening Report (PDF)",
                    data=pdf_bytes,
                    file_name=clean_filename,
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Could not compile PDF: {e}")
    else:
        st.info("👈 Please select a patient case from the sidebar or upload a fundus photograph to begin screening.")
elif screening_mode == "Clinical Bilateral Screening (OD & OS)":
    # ----------------- BILATERAL SCREENING PROTOCOL (OD vs. OS) -----------------
    if img_bgr_od is None or img_bgr_os is None:
        st.info("👈 Please select or upload fundus photographs for both the **Right Eye (OD)** and **Left Eye (OS)** in the sidebar to begin bilateral screening.")
        c_u1, c_u2 = st.columns(2)
        with c_u1:
            if img_bgr_od is not None:
                st.success(f"✅ Right Eye (OD) Loaded: `{od_source_label}`")
            else:
                st.warning("⚠️ Right Eye (OD — Oculus Dexter) scan pending.")
        with c_u2:
            if img_bgr_os is not None:
                st.success(f"✅ Left Eye (OS) Loaded: `{os_source_label}`")
            else:
                st.warning("⚠️ Left Eye (OS — Oculus Sinister) scan pending.")
    else:
        st.info(f"🔍 **Active Bilateral Workup:** Right Eye (OD): `{od_source_label}` &nbsp;|&nbsp; Left Eye (OS): `{os_source_label}`")
        img_rgb_od = cv2.cvtColor(img_bgr_od, cv2.COLOR_BGR2RGB)
        img_rgb_os = cv2.cvtColor(img_bgr_os, cv2.COLOR_BGR2RGB)

        st.subheader("1. Bilateral Image Quality Check (Automated Heuristics)")
        is_valid_od, msg_od, blur_od, bright_od = assess_image_quality(img_bgr_od, blur_threshold=blur_threshold)
        is_valid_os, msg_os, blur_os, bright_os = assess_image_quality(img_bgr_os, blur_threshold=blur_threshold)

        q1, q2 = st.columns(2)
        with q1:
            st.markdown("#### 👁️ Right Eye (OD — Oculus Dexter)")
            if is_valid_od:
                st.success(f"✅ {msg_od}")
            else:
                st.error(f"❌ {msg_od}")
            m1, m2 = st.columns(2)
            m1.metric("OD Blur Score", f"{blur_od:.1f}")
            m2.metric("OD Illumination", f"{bright_od:.1f}")

        with q2:
            st.markdown("#### 👁️ Left Eye (OS — Oculus Sinister)")
            if is_valid_os:
                st.success(f"✅ {msg_os}")
            else:
                st.error(f"❌ {msg_os}")
            m3, m4 = st.columns(2)
            m3.metric("OS Blur Score", f"{blur_os:.1f}")
            m4.metric("OS Illumination", f"{bright_os:.1f}")

        if not is_valid_od or not is_valid_os:
            st.error("🚨 **Bilateral Screening Halted:** One or both images failed clinical quality heuristics. Please re-acquire or upload clear fundus images.")
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.image(img_rgb_od, caption=f"Right Eye (OD): {'Passed' if is_valid_od else 'Rejected'}", use_container_width=True)
            with r_col2:
                st.image(img_rgb_os, caption=f"Left Eye (OS): {'Passed' if is_valid_os else 'Rejected'}", use_container_width=True)
        else:
            # 2. Run Inference & Grad-CAM++ for both eyes
            proc_rgb_od, probs_od, pred_class_od, heatmap_od = run_inference(img_bgr_od, temperature, use_tta=use_tta, multi_scale=use_multi_scale)
            proc_rgb_os, probs_os, pred_class_os, heatmap_os = run_inference(img_bgr_os, temperature, use_tta=use_tta, multi_scale=use_multi_scale)
            conf_od = float(probs_od[pred_class_od] * 100)
            conf_os = float(probs_os[pred_class_os] * 100)

            # Day 5: Bilateral CSME Macular Risk Analysis
            csme_od = detect_macular_edema_risk(img_bgr_od, heatmap=heatmap_od, dr_stage=pred_class_od)
            csme_os = detect_macular_edema_risk(img_bgr_os, heatmap=heatmap_os, dr_stage=pred_class_os)

            overall_stage = max(pred_class_od, pred_class_os)
            stage_diff = abs(pred_class_od - pred_class_os)
            higher_eye_label = "Right Eye (OD)" if pred_class_od > pred_class_os else "Left Eye (OS)" if pred_class_os > pred_class_od else "Both Eyes (Bilateral Concordance)"
            proto_overall = CLINICAL_PROTOCOLS[overall_stage]

            st.markdown("---")
            st.subheader("2. Bilateral Diagnostic Screening & Inter-Ocular Asymmetry")

            # Overall Patient Triage Card
            st.markdown(
                f"""<div style="background-color: #1E3A8A; color: white; padding: 16px 20px; border-radius: 8px; margin-bottom: 16px;">
                <h3 style="margin: 0; color: white;">🏥 OVERALL PATIENT TRIAGE: {DIAGNOSIS_INFO[overall_stage]['name'].upper()}</h3>
                <p style="margin: 6px 0 0 0; font-size: 14px; opacity: 0.95;">
                <b>Governing Eye:</b> {higher_eye_label} &nbsp;|&nbsp; 
                <b>Clinical Urgency:</b> {proto_overall['urgency']} &nbsp;|&nbsp; 
                <b>Follow-up Timeline:</b> {proto_overall['timeline']}
                </p>
                </div>""",
                unsafe_allow_html=True
            )

            # Asymmetry Assessment
            if stage_diff == 0:
                st.success(f"🟢 **Bilateral Concordance (Symmetric Involvement):** Both eyes exhibit identical microvascular severity ({DIAGNOSIS_INFO[overall_stage]['name']}). Typical systemic manifestation of diabetic retinopathy.")
            elif stage_diff == 1:
                st.warning(f"⚠️ **Mild Inter-Ocular Asymmetry (Δ = 1 Stage):** Disparity in microvascular lesion load between eyes ({higher_eye_label} at Stage {overall_stage} vs. contralateral eye at Stage {min(pred_class_od, pred_class_os)}). Triage protocol is governed strictly by the higher-severity eye.")
            else:
                st.error(f"🚨 **CLINICAL RED FLAG — MARKED INTER-OCULAR ASYMMETRY (Δ = {stage_diff} Stages):** Significant inter-ocular discrepancy detected ({higher_eye_label} is Stage {overall_stage}, while contralateral eye is Stage {min(pred_class_od, pred_class_os)}). Highly asymmetric DR is uncommon and warrants urgent carotid duplex Doppler ultrasonography (to rule out ipsilateral Ocular Ischemic Syndrome / carotid stenosis) and evaluation for asymmetric glaucoma or previous vascular compromise.")

            # Dual KPI Cards
            kpi1, kpi2 = st.columns(2)
            with kpi1:
                st.markdown(f"#### 👁️ Right Eye (OD): {DIAGNOSIS_INFO[pred_class_od]['name']}")
                st.progress(max(0, min(int(conf_od) if not np.isnan(conf_od) else 0, 100)))
                st.write(f"**Calibrated Confidence:** `{conf_od:.2f}%`")
                if conf_od < LOW_CONFIDENCE_THRESHOLD:
                    st.warning("⚠️ Low model confidence for OD. Clinician review advised.")
                st.caption(f"Suggested action: {DIAGNOSIS_INFO[pred_class_od]['action']}")
                od_csme_color = csme_od.get("badge_color", "green")
                od_badge_hex = "#EF4444" if od_csme_color == "red" else "#F59E0B" if od_csme_color == "orange" else "#10B981"
                st.markdown(f"<span style='color:{od_badge_hex}; font-weight:600;'>🎯 OD Macula: {csme_od['status']}</span> (Score: {csme_od['score']:.1f}/100 | Exudates: {csme_od['macular_exudates']} px)", unsafe_allow_html=True)

            with kpi2:
                st.markdown(f"#### 👁️ Left Eye (OS): {DIAGNOSIS_INFO[pred_class_os]['name']}")
                st.progress(max(0, min(int(conf_os) if not np.isnan(conf_os) else 0, 100)))
                st.write(f"**Calibrated Confidence:** `{conf_os:.2f}%`")
                if conf_os < LOW_CONFIDENCE_THRESHOLD:
                    st.warning("⚠️ Low model confidence for OS. Clinician review advised.")
                st.caption(f"Suggested action: {DIAGNOSIS_INFO[pred_class_os]['action']}")
                os_csme_color = csme_os.get("badge_color", "green")
                os_badge_hex = "#EF4444" if os_csme_color == "red" else "#F59E0B" if os_csme_color == "orange" else "#10B981"
                st.markdown(f"<span style='color:{os_badge_hex}; font-weight:600;'>🎯 OS Macula: {csme_os['status']}</span> (Score: {csme_os['score']:.1f}/100 | Exudates: {csme_os['macular_exudates']} px)", unsafe_allow_html=True)

            # Comparative Probability Bar Chart
            st.markdown("##### 📊 Comparative Stage Probability Distribution (OD vs. OS)")
            chart_df = pd.DataFrame({
                "Right Eye (OD) %": [round(float(probs_od[i] * 100), 2) for i in range(5)],
                "Left Eye (OS) %": [round(float(probs_os[i] * 100), 2) for i in range(5)]
            }, index=[f"Stage {i}" for i in range(5)])
            st.bar_chart(chart_df)

            # Section 3: Explainable AI (Grad-CAM++)
            st.markdown("---")
            st.subheader("3. Explainable AI (XAI) — Grad-CAM++ Dual-Eye Lesion Localization")
            st.caption("Grad-CAM++ calculates positive second- and third-order partial derivatives to isolate scattered microaneurysms, hemorrhages, and exudates across both retinas.")

            ctrl1, ctrl2, ctrl3 = st.columns([1.5, 1.2, 1.3])
            with ctrl1:
                selected_cmap = st.selectbox(
                    "🎨 Clinical Heatmap Colormap:",
                    list(COLORMAP_OPTIONS.keys()),
                    index=0,
                    key="bilateral_cmap",
                    help="Switch between classic ophthalmic JET, high-contrast Inferno, Turbo, or colorblind-accessible Viridis."
                )
            with ctrl2:
                cam_alpha = st.slider(
                    "🔎 Heatmap Opacity (α):",
                    min_value=0.10,
                    max_value=0.90,
                    value=0.45,
                    step=0.05,
                    key="bilateral_alpha",
                    help="Adjust blend ratio between retinal fundus anatomy and lesion activation map."
                )
            with ctrl3:
                st.write("")
                st.write("")
                show_contours = st.checkbox(
                    "✨ Outline High-Attention Lesions",
                    value=False,
                    key="bilateral_contours",
                    help="Draws white boundary contours around high-confidence activation zones (> 60% peak intensity)."
                )

            overlay_od = generate_cam_overlay(proc_rgb_od, heatmap_od, colormap_name=selected_cmap, alpha=cam_alpha, show_contours=show_contours)
            overlay_os = generate_cam_overlay(proc_rgb_os, heatmap_os, colormap_name=selected_cmap, alpha=cam_alpha, show_contours=show_contours)

            short_cmap = selected_cmap.split()[0]
            contour_flag = " + Contours" if show_contours else ""

            v_od, v_os = st.columns(2)
            with v_od:
                st.markdown("#### 👁️ Right Eye (OD) Workup")
                od_t1, od_t2, od_t3, od_t4 = st.tabs(["1. Original Scan", "2. Ben Graham Preprocessed", "3. Grad-CAM++ Attention", "4. Macular ROI & CSME"])
                with od_t1:
                    st.image(img_rgb_od, caption="OD: Original Retinal Scan", use_container_width=True)
                with od_t2:
                    st.image(proc_rgb_od, caption="OD: Ben Graham Enhanced (256x256)", use_container_width=True)
                with od_t3:
                    st.image(overlay_od, caption=f"OD: Grad-CAM++ ({short_cmap} | α={cam_alpha:.2f}{contour_flag})", use_container_width=True)
                with od_t4:
                    st.image(csme_od["annotated_rgb"], caption=f"OD Macula: {csme_od['status']} ({csme_od['score']:.1f}/100)", use_container_width=True)

            with v_os:
                st.markdown("#### 👁️ Left Eye (OS) Workup")
                os_t1, os_t2, os_t3, os_t4 = st.tabs(["1. Original Scan", "2. Ben Graham Preprocessed", "3. Grad-CAM++ Attention", "4. Macular ROI & CSME"])
                with os_t1:
                    st.image(img_rgb_os, caption="OS: Original Retinal Scan", use_container_width=True)
                with os_t2:
                    st.image(proc_rgb_os, caption="OS: Ben Graham Enhanced (256x256)", use_container_width=True)
                with os_t3:
                    st.image(overlay_os, caption=f"OS: Grad-CAM++ ({short_cmap} | α={cam_alpha:.2f}{contour_flag})", use_container_width=True)
                with os_t4:
                    st.image(csme_os["annotated_rgb"], caption=f"OS Macula: {csme_os['status']} ({csme_os['score']:.1f}/100)", use_container_width=True)

            # Section 4: Clinical Care Pathway & Specialist Referral
            st.markdown("---")
            st.subheader("4. Recommended Clinical Care Pathway & Specialist Referral")
            c_triage, c_timeline = st.columns([1.2, 1])
            with c_triage:
                if overall_stage >= 2:
                    st.error(f"🚨 **Overall Triage Priority:** {proto_overall['urgency']}")
                elif overall_stage == 1:
                    st.warning(f"⚠️ **Overall Triage Priority:** {proto_overall['urgency']}")
                else:
                    st.success(f"🟢 **Overall Triage Priority:** {proto_overall['urgency']}")
                st.write(f"**Referral Care Level:** `{proto_overall['referral_level']}`")
                st.write(f"**Diagnostic Workup:** {proto_overall['workup']}")

            with c_timeline:
                st.info(f"⏱️ **Follow-up Timeline:** {proto_overall['timeline']}")
                if p_hba1c >= 8.0:
                    st.warning(f"⚠️ Patient HbA1c is **{p_hba1c:.1f}%** (elevated). Tight glycemic control with an endocrinologist is critical to prevent DR acceleration.")

            # Evidence-Based Systemic & Lifestyle Guidance
            with st.expander("📋 Evidence-Based Clinical Management Guidelines (ICO/WHO Standards)", expanded=True):
                st.write(f"**Clinical Summary:** {proto_overall['summary']}")
                st.write("**Key Management Goals:**")
                for goal in proto_overall['target_goals']:
                    st.markdown(f"- {goal}")
                st.caption("ℹ️ Guidance aligned with International Council of Ophthalmology (ICO) Diabetic Eye Care Guidelines.")

            # Nearby Eye Hospital Locator
            loc_clean = "Hyderabad"
            with st.expander("📍 Find Nearby Eye Hospitals & Retina Specialists (Primary Care Triage)", expanded=(overall_stage >= 2)):
                st.markdown("For patients identified with referable or sight-threatening diabetic retinopathy, prompt referral to an accredited vitreo-retinal facility is essential.")
                loc_col1, loc_col2 = st.columns([2.5, 1])
                with loc_col1:
                    user_loc = st.text_input(
                        "Enter Patient's City, Area, or PIN Code:",
                        value="Hyderabad",
                        key="bilateral_loc_input",
                        help="Enter any town, area, or city (e.g., Medchal, Secunderabad, Warangal, Karimnagar, Bangalore)."
                    )
                with loc_col2:
                    st.write("")
                    st.write("")
                    loc_clean = user_loc.strip() if user_loc and user_loc.strip() else "near me"

                encoded_loc = urllib.parse.quote_plus(f"retina specialist eye hospital in {loc_clean}")
                maps_url = f"https://www.google.com/maps/search/{encoded_loc}"
                st.link_button(
                    f"🔎 Search Eye Hospitals & Retina Specialists in '{loc_clean}' (Google Maps)",
                    url=maps_url,
                    use_container_width=True
                )
                st.markdown("#### Accredited Tertiary Eye Care Centers (Directory):")
                for region, hospitals in EYE_HOSPITALS_DIRECTORY.items():
                    st.markdown(f"**{region}:**")
                    for h in hospitals:
                        st.markdown(f"- **[{h['name']}]({h['url']})** — *{h['city']}* ({h['type']}) | 📞 `{h['contact']}`")

            # Section 5: Bilateral Clinical Notes & PDF Report
            st.markdown("---")
            st.subheader("5. Bilateral Clinical Notes & Unified PDF Report")

            patient_info_dict = {
                "name": p_name,
                "age": p_age,
                "gender": p_gender,
                "duration": p_duration,
                "hba1c": p_hba1c,
                "location": loc_clean
            }

            notes_text, is_gemini, notes_error = generate_bilateral_clinical_notes(
                patient_info=patient_info_dict,
                od_stage=pred_class_od,
                od_conf=conf_od,
                os_stage=pred_class_os,
                os_conf=conf_os,
                api_key=gemini_key.strip() if gemini_key else None,
                od_csme=csme_od,
                os_csme=csme_os
            )

            if notes_error:
                st.warning(f"ℹ️ {notes_error}")
            if is_gemini:
                st.success("🤖 Bilateral notes drafted by **Gemini (gemini-2.5-flash)** from the screening result only. Review before use.")
            else:
                st.caption("ℹ️ Bilateral notes generated by the built-in offline clinical engine.")

            st.markdown(notes_text)

            st.markdown("#### 📄 Export Bilateral Screening Report")
            try:
                pdf_bytes = generate_bilateral_pdf_report(
                    patient_info=patient_info_dict,
                    od_stage_idx=pred_class_od,
                    od_confidence=conf_od,
                    od_orig_rgb=cv2.resize(img_rgb_od, (300, 300)),
                    od_cam_rgb=cv2.resize(overlay_od, (300, 300)),
                    od_blur=blur_od,
                    os_stage_idx=pred_class_os,
                    os_confidence=conf_os,
                    os_orig_rgb=cv2.resize(img_rgb_os, (300, 300)),
                    os_cam_rgb=cv2.resize(overlay_os, (300, 300)),
                    os_blur=blur_os,
                    clinical_notes_text=notes_text,
                    od_csme=csme_od,
                    os_csme=csme_os
                )

                safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', str(p_name).strip()) or "Patient"
                clean_filename = f"Retina_Bilateral_Report_{safe_name}_{datetime.now().strftime('%Y%m%d')}.pdf"
                st.download_button(
                    label="📥 Download Bilateral Screening Report (PDF)",
                    data=pdf_bytes,
                    file_name=clean_filename,
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Could not compile Bilateral PDF: {e}")
elif screening_mode == "Screening Camp Batch Processing (Bulk Triage)":
    st.subheader("⛺ Screening Camp Batch Processing & Triage (Epidemiological Workup)")
    st.caption("High-throughput automated screening for community vision camps, mobile health vans, and rural outreach programs.")

    camp_meta = {
        "camp_name": camp_name,
        "location": camp_loc,
        "screener_id": screener_id
    }

    st.markdown(
        f"""<div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 8px; padding: 12px 18px; margin-bottom: 16px;">
        <span style="font-size: 15px; font-weight: 600; color: #1E3A8A;">🏥 Facility: {html.escape(camp_name)}</span> &nbsp;|&nbsp; 
        <span style="font-size: 14px; color: #475569;">📍 Location: {html.escape(camp_loc)}</span> &nbsp;|&nbsp; 
        <span style="font-size: 14px; color: #475569;">👤 Screener ID: {html.escape(screener_id)}</span> &nbsp;|&nbsp; 
        <span style="font-size: 14px; font-weight: 600; color: #0284C7;">📁 Loaded Scans: {len(batch_scans_data)}</span>
        </div>""",
        unsafe_allow_html=True
    )

    if not batch_scans_data:
        st.warning("👈 Please select or upload fundus scans in the sidebar to populate the screening batch.")
    else:
        btn_col1, btn_col2 = st.columns([2, 1])
        with btn_col1:
            run_bulk = st.button("🚀 Run Automated Quality Check, TTA Inference & CSME Triage", type="primary", use_container_width=True)
        with btn_col2:
            if st.button("🔄 Reset / Clear Batch Cache", use_container_width=True):
                if "batch_triage_results" in st.session_state:
                    del st.session_state["batch_triage_results"]
                st.rerun()

        if run_bulk or "batch_triage_results" in st.session_state:
            if run_bulk or ("batch_triage_results" not in st.session_state):
                batch_results = []
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                for idx, (scan_id, scan_img) in enumerate(batch_scans_data):
                    status_text.text(f"Processing scan {idx+1}/{len(batch_scans_data)}: {scan_id}")
                    is_valid, q_msg, blur_v, bright_v = assess_image_quality(scan_img, blur_threshold=blur_threshold)

                    if not is_valid:
                        rec = {
                            "Scan_ID": scan_id,
                            "Filename": scan_id,
                            "Quality_Status": "Rejected",
                            "Quality_Notes": q_msg,
                            "Blur_Score": round(float(blur_v), 1),
                            "Brightness": round(float(bright_v), 1),
                            "DR_Stage_Index": -1,
                            "DR_Stage": "Unusable",
                            "Confidence_%": 0.0,
                            "Referable_DR": "Re-scan Required",
                            "CSME_Risk": "Not Evaluated",
                            "CSME_Score": 0.0,
                            "Triage_Urgency": "Immediate Re-scan Required",
                            "Follow_up_Timeline": "Immediate",
                            "Recommended_Action": "Immediate camera re-acquisition required on site",
                            "_img_rgb": cv2.cvtColor(scan_img, cv2.COLOR_BGR2RGB),
                            "_proc_rgb": None,
                            "_overlay_rgb": None,
                            "_csme_rgb": None,
                            "_overlay_buf": None
                        }
                    else:
                        proc_rgb, probs, pred_c, heatmap = run_inference(scan_img, temperature, use_tta=use_tta, multi_scale=use_multi_scale)
                        conf = float(probs[pred_c] * 100)
                        csme_res = detect_macular_edema_risk(scan_img, heatmap=heatmap, dr_stage=pred_c)

                        overlay = generate_cam_overlay(proc_rgb, heatmap, colormap_name="JET (Classic Ophthalmology)", alpha=0.45)
                        is_ok, buf = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                        overlay_bytes = buf.tobytes() if is_ok else None

                        is_ref = "Yes (Referable)" if (pred_c >= 2 or csme_res["key"] in ["Moderate", "High"]) else "No"
                        proto = CLINICAL_PROTOCOLS[pred_c]

                        rec = {
                            "Scan_ID": scan_id,
                            "Filename": scan_id,
                            "Quality_Status": "Passed",
                            "Quality_Notes": q_msg,
                            "Blur_Score": round(float(blur_v), 1),
                            "Brightness": round(float(bright_v), 1),
                            "DR_Stage_Index": int(pred_c),
                            "DR_Stage": DIAGNOSIS_INFO[pred_c]["name"],
                            "Confidence_%": round(conf, 1),
                            "Referable_DR": is_ref,
                            "CSME_Risk": csme_res["name"],
                            "CSME_Score": round(float(csme_res["score"]), 1),
                            "Triage_Urgency": proto["urgency"],
                            "Follow_up_Timeline": proto["timeline"],
                            "Recommended_Action": f"{proto['urgency']} — {proto['referral_level']}",
                            "_img_rgb": cv2.cvtColor(scan_img, cv2.COLOR_BGR2RGB),
                            "_proc_rgb": proc_rgb,
                            "_overlay_rgb": overlay,
                            "_csme_rgb": csme_res["annotated_rgb"],
                            "_overlay_buf": overlay_bytes
                        }
                    batch_results.append(rec)
                    progress_bar.progress((idx + 1) / len(batch_scans_data))

                status_text.text("Batch processing complete.")
                st.session_state["batch_triage_results"] = batch_results

            results = st.session_state["batch_triage_results"]

            # Epidemiological KPI metrics
            total_n = len(results)
            passed_n = sum(1 for r in results if r["Quality_Status"] == "Passed")
            rejected_n = total_n - passed_n
            referable_n = sum(1 for r in results if "Yes" in str(r.get("Referable_DR", "")))
            urgent_n = sum(1 for r in results if r.get("DR_Stage_Index") in [3, 4])
            csme_high_n = sum(1 for r in results if any(k in str(r.get("CSME_Risk", "")) for k in ["High", "Moderate"]))

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Total Scanned", total_n)
            m2.metric("Quality Passed", f"{passed_n} ({passed_n/max(total_n,1)*100:.0f}%)")
            m3.metric("Re-scan Rejections", f"{rejected_n} ({rejected_n/max(total_n,1)*100:.0f}%)")
            m4.metric("Referable DR", f"{referable_n} ({referable_n/max(total_n,1)*100:.0f}%)")
            m5.metric("Sight-Threatening", f"{urgent_n} ({urgent_n/max(total_n,1)*100:.0f}%)")
            m6.metric("CSME Risk Alert", f"{csme_high_n} ({csme_high_n/max(total_n,1)*100:.0f}%)")

            # Population Distribution Charts
            st.markdown("---")
            st.subheader("📊 Epidemiological & Risk Stratification Charts")

            chart_c1, chart_c2 = st.columns(2)
            with chart_c1:
                st.markdown("##### Retinopathy Severity Distribution (DR Stage)")
                stage_labels = ["Stage 0: No DR", "Stage 1: Mild", "Stage 2: Moderate", "Stage 3: Severe", "Stage 4: Proliferative", "Quality Rejected"]
                stage_counts = [
                    sum(1 for r in results if r.get("DR_Stage_Index") == 0),
                    sum(1 for r in results if r.get("DR_Stage_Index") == 1),
                    sum(1 for r in results if r.get("DR_Stage_Index") == 2),
                    sum(1 for r in results if r.get("DR_Stage_Index") == 3),
                    sum(1 for r in results if r.get("DR_Stage_Index") == 4),
                    sum(1 for r in results if r.get("DR_Stage_Index") == -1)
                ]
                df_stage_chart = pd.DataFrame({"Patients": stage_counts}, index=stage_labels)
                st.bar_chart(df_stage_chart)

            with chart_c2:
                st.markdown("##### Macular Edema (CSME) Threat Distribution")
                csme_labels = ["Low / Negative", "Moderate / Pre-CSME", "High CSME Risk", "Quality Rejected"]
                csme_counts = [
                    sum(1 for r in results if "Low" in str(r.get("CSME_Risk", ""))),
                    sum(1 for r in results if "Moderate" in str(r.get("CSME_Risk", ""))),
                    sum(1 for r in results if "High" in str(r.get("CSME_Risk", ""))),
                    sum(1 for r in results if r.get("DR_Stage_Index") == -1)
                ]
                df_csme_chart = pd.DataFrame({"Patients": csme_counts}, index=csme_labels)
                st.bar_chart(df_csme_chart)

            # Filterable Triage Manifest
            st.markdown("---")
            st.subheader("📋 Screening Camp Patient Manifest & Triage Register")

            filter_opt = st.selectbox(
                "Filter Cohort by Clinical Triage Category:",
                [
                    "All Patients (Complete Manifest)",
                    "Referable DR Only (Stage >= 2 or CSME)",
                    "Urgent / Sight-Threatening DR (Stage 3 & 4)",
                    "Macular CSME Alert Only (Moderate / High)",
                    "Quality Rejected Scans Only (Need Re-acquisition)"
                ]
            )

            if filter_opt == "Referable DR Only (Stage >= 2 or CSME)":
                filtered_results = [r for r in results if "Yes" in str(r.get("Referable_DR", ""))]
            elif filter_opt == "Urgent / Sight-Threatening DR (Stage 3 & 4)":
                filtered_results = [r for r in results if r.get("DR_Stage_Index") in [3, 4]]
            elif filter_opt == "Macular CSME Alert Only (Moderate / High)":
                filtered_results = [r for r in results if any(w in str(r.get("CSME_Risk", "")) for w in ["Moderate", "High"])]
            elif filter_opt == "Quality Rejected Scans Only (Need Re-acquisition)":
                filtered_results = [r for r in results if r.get("Quality_Status") == "Rejected"]
            else:
                filtered_results = results

            clean_display_data = [{k: v for k, v in r.items() if not k.startswith('_')} for r in filtered_results]
            df_display = pd.DataFrame(clean_display_data)
            st.dataframe(df_display, use_container_width=True)
            st.caption(f"Showing {len(filtered_results)} of {total_n} cases.")

            # Collapsible Visual Patient Gallery
            with st.expander(f"👁️ View Case Scans & XAI Lesion Overlays ({len(filtered_results)} Filtered Cases)", expanded=False):
                st.write("Browse original retinal scans, Grad-CAM++ lesion heatmaps, and macular edema (CSME) threat zones.")
                for item in filtered_results:
                    st.markdown(f"#### 📄 `{item.get('Scan_ID')}` — {item.get('DR_Stage')} | {item.get('CSME_Risk')}")
                    if item.get("Quality_Status") == "Passed" and item.get("_img_rgb") is not None:
                        g_col1, g_col2, g_col3 = st.columns(3)
                        with g_col1:
                            st.image(item["_img_rgb"], caption="1. Original Scan", use_container_width=True)
                        with g_col2:
                            if item.get("_overlay_rgb") is not None:
                                st.image(item["_overlay_rgb"], caption=f"2. Grad-CAM++ ({item['DR_Stage']})", use_container_width=True)
                        with g_col3:
                            if item.get("_csme_rgb") is not None:
                                st.image(item["_csme_rgb"], caption=f"3. Macular Zones & CSME ({item['CSME_Risk']})", use_container_width=True)
                    else:
                        st.warning(f"Scan failed automated quality check: {item.get('Quality_Notes', 'Quality Check Failed')}. Re-acquisition required.")
                        if item.get("_img_rgb") is not None:
                            st.image(item["_img_rgb"], caption="Rejected Scan (Failed Quality Check)", width=350)
                    st.markdown("---")

            # Export
            st.markdown("---")
            st.markdown("#### 📦 Export Camp Manifest & Complete ZIP Package")
            csv_bytes, zip_bytes = generate_camp_batch_archive(results, camp_meta)
            exp_c1, exp_c2 = st.columns(2)
            with exp_c1:
                st.download_button(
                    label="📥 Download Camp Manifest (CSV)",
                    data=csv_bytes,
                    file_name=f"Camp_Manifest_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with exp_c2:
                st.download_button(
                    label="📦 Download Complete Camp Package (ZIP: CSV + Summary + Grad-CAM Overlays)",
                    data=zip_bytes,
                    file_name=f"Camp_Triage_Package_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
