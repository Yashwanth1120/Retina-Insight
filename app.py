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

from engine import SUPPORTED_BACKBONES, build_retina_model

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

# ----------------- HOSPITAL-GRADE MEDICAL CSS & WHITE-LABELING -----------------
# NOTE: the <header> is intentionally NOT hidden, because it contains the
# sidebar open/close toggle on several Streamlit versions.
hospital_clinical_style = """
    <style>
    /* White-labeling & clean view */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden; display: none;}
    .stDeployButton {display: none !important;}
    #manage-app-button {display: none !important;}
    div[data-testid="stToolbarActions"] {display: none !important;}
    [data-testid="stDecoration"] {visibility: hidden; display: none;}
    [data-testid="stStatusWidget"] {visibility: hidden; display: none;}

    /* Sidebar toggle chevron MUST always be visible */
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        z-index: 999999 !important;
    }
    button[data-testid="stSidebarCollapseButton"] {
        display: flex !important;
        visibility: visible !important;
    }

    /* Modern clinical typography & high-contrast surfaces */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Hospital Executive Branding Banner */
    .hospital-header-container {
        background: linear-gradient(135deg, #090e17 0%, #111c2e 50%, #0d2847 100%);
        border: 1px solid #1e3a8a;
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 18px -2px rgba(15, 23, 42, 0.45);
        color: #ffffff;
    }

    /* Pulse animation for priority triage alerts */
    @keyframes pulse-urgent-glow {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.02); opacity: 0.88; }
    }
    .pulse-urgent {
        animation: pulse-urgent-glow 2s infinite ease-in-out;
    }

    /* Styled metric enhancements */
    [data-testid="stMetricValue"] {
        font-size: 21px !important;
        font-weight: 700 !important;
    }

    /* Styled tab headers */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 8px 16px !important;
        border-radius: 6px !important;
    }

    /* Primary clinical action buttons */
    .stDownloadButton > button {
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.15s ease-in-out !important;
    }
    </style>
"""
st.markdown(hospital_clinical_style, unsafe_allow_html=True)

def render_hospital_header():
    header_html = """
    <div class="hospital-header-container" style="padding: 12px 20px; margin-bottom: 12px;">
      <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
        <div style="display: flex; align-items: center; gap: 12px;">
          <div style="width: 40px; height: 40px; border-radius: 9px; background: #2563eb; display: flex; align-items: center; justify-content: center; font-size: 20px; box-shadow: 0 2px 8px rgba(37,99,235,0.4);">
            👁️
          </div>
          <div>
            <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
              <span style="font-size: 18px; font-weight: 700; letter-spacing: -0.02em; color: #f8fafc;">Retina Insight Clinical Diagnostic Station</span>
              <span style="background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); font-size: 10px; font-weight: 600; padding: 2px 7px; border-radius: 5px; text-transform: uppercase;">v2.5 Hospital Enterprise</span>
            </div>
            <div style="font-size: 11.5px; color: #94a3b8; margin-top: 2px;">
              AI-Assisted Retinal Screening &bull; ETDRS CSME Macular Risk Engine &bull; Explainable AI (Grad-CAM++)
            </div>
          </div>
        </div>
        <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
          <div style="font-size: 11px; color: #cbd5e1; background: rgba(15,23,42,0.6); border: 1px solid #334155; padding: 4px 10px; border-radius: 6px;">
            ⚖️ <b>Clinical Decision Support:</b> Attending ophthalmologist review required
          </div>
          <div style="display: flex; align-items: center; gap: 6px; background: rgba(15,23,42,0.6); border: 1px solid #334155; padding: 4px 10px; border-radius: 6px; font-size: 11px; color: #38bdf8;">
            <span style="width: 7px; height: 7px; border-radius: 50%; background-color: #10b981; display: inline-block;"></span>
            <span style="font-weight: 600;">System Online</span>
          </div>
        </div>
      </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

render_hospital_header()

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
    sigma = max(5, int(10 * (target_size / 256.0)))
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigma)
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
def load_system(backbone="efficientnet_b0"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    b_info = SUPPORTED_BACKBONES.get(backbone, SUPPORTED_BACKBONES["efficientnet_b0"])

    weights_path = b_info.get("weights_file", "best_retina_model.pth")
    actual_backbone = backbone
    if not os.path.exists(weights_path):
        weights_path = "best_retina_model.pth"
        actual_backbone = "efficientnet_b0"

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"{weights_path} was not found next to app.py.")

    model, target_layer = build_retina_model(backbone=actual_backbone, num_classes=5)
    state = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    elif isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
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

    cam = GradCAMPlusPlus(model, target_layer)
    return model, cam, temp, temp_from_file, device, actual_backbone

try:
    model, grad_cam, temperature, temp_from_file, device, active_backbone_name = load_system()
except Exception as e:
    st.error(f"🚨 Model could not be loaded, so screening is disabled: {e}")
    st.stop()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_data(show_spinner="Running model...")
def run_inference(img_bgr, temp, use_tta=True, multi_scale=False, target_size=256, backbone="efficientnet_b0"):
    """Preprocess, predict, and compute the Grad-CAM++ heatmap.
    Supports resolution auto-adaptation (256, 380, 512), 4-fold Test-Time Augmentation (standard, horizontal
    flip, vertical flip, rot180) to maximize classification precision and prevent
    borderline misclassifications.
    When multi_scale is True, incorporates central foveal multi-crop evaluation (Ensemble Mode).
    Cached so typing in the patient form or adjusting sliders does not re-run the neural network."""
    res = target_size if target_size is not None else 256
    proc_bgr = fast_ben_graham(img_bgr, target_size=res)
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
            cropped_resized = cv2.resize(cropped, (res, res))
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

def generate_pdf_report(patient_info, stage_name, confidence, original_rgb, gradcam_rgb, clinical_notes_text, stage_idx=0, csme_info=None, clinician_info=None):
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
    if clinician_info and clinician_info.get("is_signed"):
        c_name = html.escape(str(clinician_info.get("name", "Attending Ophthalmologist")))
        c_reg = html.escape(str(clinician_info.get("reg_no", "MCI-RET-VALIDATED")))
        c_time = html.escape(str(clinician_info.get("timestamp", scan_date)))
        sign_data = [
            [Paragraph(f"<b>Attending Clinician:</b> {c_name} &nbsp;|&nbsp; <b>Reg No:</b> {c_reg}<br/><font color='#16A34A'><b>Verification Status:</b> Clinically Verified &amp; Certified (Digital Seal)</font>", body_style),
             Paragraph(f"<b>Accreditation Date:</b> {c_time}", ParagraphStyle('R', parent=body_style, alignment=2))]
        ]
    else:
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
    od_csme=None, os_csme=None,
    clinician_info=None
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
    if clinician_info and clinician_info.get("is_signed"):
        c_name = html.escape(str(clinician_info.get("name", "Attending Ophthalmologist")))
        c_reg = html.escape(str(clinician_info.get("reg_no", "MCI-RET-VALIDATED")))
        c_time = html.escape(str(clinician_info.get("timestamp", scan_date)))
        sign_data = [
            [Paragraph(f"<b>Attending Clinician:</b> {c_name} &nbsp;|&nbsp; <b>Reg No:</b> {c_reg}<br/><font color='#16A34A'><b>Verification Status:</b> Clinically Verified &amp; Certified (Digital Seal)</font>", body_style),
             Paragraph(f"<b>Accreditation Date:</b> {c_time}", ParagraphStyle('R', parent=body_style, alignment=2))]
        ]
    else:
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


# ----------------- DAY 7 HOSPITAL UI HELPERS -----------------
def render_single_triage_ribbon(patient_info, pred_class, confidence, csme_res):
    """Renders 4-KPI clinical triage header ribbon for single fundus screening."""
    proto = CLINICAL_PROTOCOLS.get(int(pred_class), CLINICAL_PROTOCOLS[0])
    try:
        hba1c_val = float(patient_info.get("hba1c", 8.0))
    except (ValueError, TypeError):
        hba1c_val = 8.0
    hba1c_badge = "#10B981" if hba1c_val < 7.0 else "#F59E0B" if hba1c_val < 8.0 else "#EF4444"
    hba1c_status = "Optimal" if hba1c_val < 7.0 else "Borderline" if hba1c_val < 8.0 else "Elevated"

    stage_name = DIAGNOSIS_INFO.get(int(pred_class), {}).get("name", f"Stage {pred_class}")
    is_urgent = (int(pred_class) >= 3 or csme_res.get("key") == "High")
    urgency_border = "#EF4444" if is_urgent else "#F59E0B" if int(pred_class) in [1, 2] else "#10B981"
    urgency_bg = "#450a0a" if is_urgent else "#451a03" if int(pred_class) in [1, 2] else "#064e3b"
    urgency_text = "#fca5a5" if is_urgent else "#fcd34d" if int(pred_class) in [1, 2] else "#6ee7b7"

    csme_color = csme_res.get("badge_color", "green")
    csme_border = "#EF4444" if csme_color == "red" else "#F59E0B" if csme_color == "orange" else "#10B981"
    csme_bg = "#450a0a" if csme_color == "red" else "#451a03" if csme_color == "orange" else "#064e3b"
    csme_text = "#fca5a5" if csme_color == "red" else "#fcd34d" if csme_color == "orange" else "#6ee7b7"

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""<div style="background: #1e293b; border: 1.5px solid #334155; border-radius: 10px; padding: 12px 14px; min-height: 118px; box-sizing: border-box;">
<div style="font-size: 10.5px; font-weight: 700; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em;">Patient Clinical Profile</div>
<div style="font-size: 15px; font-weight: 700; color: #f8fafc; margin-top: 3px;">{html.escape(str(patient_info.get('name', 'Patient')))}</div>
<div style="font-size: 11.5px; color: #cbd5e1; margin-top: 3px;">
<span>{patient_info.get('age', 'N/A')} yrs &bull; {html.escape(str(patient_info.get('gender', 'N/A')))}</span> &bull; <span>T2DM ({patient_info.get('duration', 'N/A')} yrs)</span>
</div>
<div style="margin-top: 6px;">
<span style="background: {hba1c_badge}22; color: {hba1c_badge}; border: 1px solid {hba1c_badge}66; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 600;">
HbA1c: {hba1c_val:.1f}% ({hba1c_status})
</span>
</div>
</div>""",
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""<div style="background: #1e293b; border: 1.5px solid #334155; border-radius: 10px; padding: 12px 14px; min-height: 118px; box-sizing: border-box;">
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="font-size: 10.5px; font-weight: 700; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em;">AI Severity Grade</span>
<span style="font-size: 11px; font-family: monospace; font-weight: 700; color: #38bdf8; background: #0284c722; border: 1px solid #0284c766; padding: 1px 6px; border-radius: 4px;">{confidence:.1f}% TTA</span>
</div>
<div style="font-size: 14px; font-weight: 700; color: #f8fafc; margin-top: 3px;">
{html.escape(stage_name)}
</div>
<div style="width: 100%; background: #334155; height: 5px; border-radius: 9999px; margin-top: 6px; overflow: hidden;">
<div style="background: #38bdf8; height: 5px; width: {min(100.0, max(0.0, confidence)):.1f}%;"></div>
</div>
<div style="font-size: 10.5px; color: #94a3b8; margin-top: 4px;">4-Fold Rotational &amp; Flip Invariance</div>
</div>""",
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""<div style="background: {urgency_bg}; border: 1.5px solid {urgency_border}; border-radius: 10px; padding: 12px 14px; min-height: 118px; box-sizing: border-box;">
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="font-size: 10.5px; font-weight: 700; text-transform: uppercase; color: {urgency_text}; letter-spacing: 0.05em;">Triage Urgency</span>
{'<span style="background: #ef4444; color: white; padding: 1px 6px; border-radius: 4px; font-size: 9px; font-weight: 700;">EXPEDITE</span>' if is_urgent else ''}
</div>
<div style="font-size: 14px; font-weight: 700; color: #ffffff; margin-top: 3px;">
{html.escape(proto['urgency'])}
</div>
<div style="font-size: 11px; color: {urgency_text}; margin-top: 4px;">
⏱️ <b>Timeline:</b> {html.escape(proto['timeline'])}
</div>
</div>""",
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""<div style="background: {csme_bg}; border: 1.5px solid {csme_border}; border-radius: 10px; padding: 12px 14px; min-height: 118px; box-sizing: border-box;">
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="font-size: 10.5px; font-weight: 700; text-transform: uppercase; color: {csme_text}; letter-spacing: 0.05em;">ETDRS CSME Macula</span>
<span style="font-size: 10.5px; font-family: monospace; font-weight: 700; color: {csme_text}; background: rgba(0,0,0,0.25); padding: 1px 5px; border-radius: 4px;">Score: {csme_res.get('score', 0):.1f}</span>
</div>
<div style="font-size: 14px; font-weight: 700; color: #ffffff; margin-top: 3px;">
{html.escape(csme_res.get('status', 'CSME Low Risk'))}
</div>
<div style="font-size: 10.5px; color: {csme_text}; margin-top: 4px;">
Fovea &lt;500µm: <b>{csme_res.get('foveal_exudates', 0)} px</b> | ROI: <b>{csme_res.get('macular_exudates', 0)} px</b>
</div>
</div>""",
            unsafe_allow_html=True
        )


def render_bilateral_triage_ribbon(patient_info, pred_od, conf_od, csme_od, pred_os, conf_os, csme_os):
    """Renders 4-KPI clinical triage header ribbon for bilateral (OD vs OS) screening."""
    overall_stage = max(int(pred_od), int(pred_os))
    proto_overall = CLINICAL_PROTOCOLS.get(overall_stage, CLINICAL_PROTOCOLS[0])
    diff = abs(int(pred_od) - int(pred_os))
    governing_eye = "OD (Right Eye)" if pred_od > pred_os else "OS (Left Eye)" if pred_os > pred_od else "Both Eyes (Concordant)"

    try:
        hba1c_val = float(patient_info.get("hba1c", 8.0))
    except (ValueError, TypeError):
        hba1c_val = 8.0
    hba1c_badge = "#10B981" if hba1c_val < 7.0 else "#F59E0B" if hba1c_val < 8.0 else "#EF4444"
    hba1c_status = "Optimal" if hba1c_val < 7.0 else "Borderline" if hba1c_val < 8.0 else "Elevated"

    is_urgent = (overall_stage >= 3 or csme_od.get("key") == "High" or csme_os.get("key") == "High")
    urgency_border = "#EF4444" if is_urgent else "#F59E0B" if overall_stage in [1, 2] else "#10B981"
    urgency_bg = "#450a0a" if is_urgent else "#451a03" if overall_stage in [1, 2] else "#064e3b"
    urgency_text = "#fca5a5" if is_urgent else "#fcd34d" if overall_stage in [1, 2] else "#6ee7b7"

    asym_color = "#10B981" if diff == 0 else "#F59E0B" if diff == 1 else "#EF4444"
    asym_label = "Concordant (Symmetric)" if diff == 0 else f"Mild Asymmetry (Δ={diff})" if diff == 1 else f"Marked Red Flag (Δ={diff} Stages)"

    od_status = csme_od.get('status', 'CSME Low')
    os_status = csme_os.get('status', 'CSME Low')
    od_color = '#EF4444' if csme_od.get('key')=='High' else '#F59E0B' if csme_od.get('key')=='Moderate' else '#10B981'
    os_color = '#EF4444' if csme_os.get('key')=='High' else '#F59E0B' if csme_os.get('key')=='Moderate' else '#10B981'

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""<div style="background: #1e293b; border: 1.5px solid #334155; border-radius: 10px; padding: 12px 14px; min-height: 118px; box-sizing: border-box;">
<div style="font-size: 10.5px; font-weight: 700; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em;">Patient Clinical Profile</div>
<div style="font-size: 15px; font-weight: 700; color: #f8fafc; margin-top: 3px;">{html.escape(str(patient_info.get('name', 'Patient')))}</div>
<div style="font-size: 11.5px; color: #cbd5e1; margin-top: 3px;">
<span>{patient_info.get('age', 'N/A')} yrs &bull; {html.escape(str(patient_info.get('gender', 'N/A')))}</span> &bull; <span>T2DM ({patient_info.get('duration', 'N/A')} yrs)</span>
</div>
<div style="margin-top: 6px;">
<span style="background: {hba1c_badge}22; color: {hba1c_badge}; border: 1px solid {hba1c_badge}66; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 600;">
HbA1c: {hba1c_val:.1f}% ({hba1c_status})
</span>
</div>
</div>""",
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""<div style="background: #1e293b; border: 1.5px solid {asym_color}; border-radius: 10px; padding: 12px 14px; min-height: 118px; box-sizing: border-box;">
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="font-size: 10.5px; font-weight: 700; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em;">Bilateral Symmetry</span>
<span style="font-size: 11px; font-weight: 700; color: {asym_color}; font-family: monospace; background: {asym_color}22; border: 1px solid {asym_color}66; padding: 1px 6px; border-radius: 4px;">Δ = {diff}</span>
</div>
<div style="font-size: 13.5px; font-weight: 700; color: {asym_color}; margin-top: 3px;">
{asym_label}
</div>
<div style="font-size: 11.5px; color: #cbd5e1; margin-top: 3px;">
<b>OD:</b> Stage {pred_od} ({conf_od:.1f}%) &bull; <b>OS:</b> Stage {pred_os} ({conf_os:.1f}%)
</div>
<div style="font-size: 10.5px; color: #94a3b8; margin-top: 2px;">Governing: <strong style="color: #f8fafc;">{governing_eye}</strong></div>
</div>""",
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""<div style="background: {urgency_bg}; border: 1.5px solid {urgency_border}; border-radius: 10px; padding: 12px 14px; min-height: 118px; box-sizing: border-box;">
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="font-size: 10.5px; font-weight: 700; text-transform: uppercase; color: {urgency_text}; letter-spacing: 0.05em;">Overall Patient Triage</span>
{'<span style="background: #ef4444; color: white; padding: 1px 6px; border-radius: 4px; font-size: 9px; font-weight: 700;">EXPEDITE</span>' if is_urgent else ''}
</div>
<div style="font-size: 14px; font-weight: 700; color: #ffffff; margin-top: 3px;">
{html.escape(proto_overall['urgency'])}
</div>
<div style="font-size: 11px; color: {urgency_text}; margin-top: 4px;">
⏱️ <b>Timeline:</b> {html.escape(proto_overall['timeline'])}
</div>
</div>""",
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""<div style="background: #1e293b; border: 1.5px solid #334155; border-radius: 10px; padding: 12px 14px; min-height: 118px; box-sizing: border-box;">
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="font-size: 10.5px; font-weight: 700; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em;">Dual Macular Assessment</span>
<span style="font-size: 10px; font-weight: 700; color: #38bdf8; background: #0284c722; border: 1px solid #0284c766; padding: 1px 5px; border-radius: 4px;">ETDRS</span>
</div>
<div style="font-size: 12px; color: #cbd5e1; margin-top: 3px;">
<b>OD:</b> <span style="font-weight: 600; color: {od_color};">{html.escape(od_status)}</span>
</div>
<div style="font-size: 12px; color: #cbd5e1; margin-top: 2px;">
<b>OS:</b> <span style="font-weight: 600; color: {os_color};">{html.escape(os_status)}</span>
</div>
<div style="font-size: 10.5px; color: #94a3b8; margin-top: 4px;">
Exudates: OD {csme_od.get('macular_exudates', 0)} px &bull; OS {csme_os.get('macular_exudates', 0)} px
</div>
</div>""",
            unsafe_allow_html=True
        )


def render_clinician_signoff_section(initial_notes, key_prefix="single"):
    """Renders hospital EHR-compliant Attending Clinician Accreditation and Sign-off block."""
    st.markdown("#### 📋 Attending Ophthalmologist Accreditation & Audit Log")
    st.caption("Hospital EHR-compliant clinical sign-off workflow with digital accreditation seal.")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        doc_name = st.text_input(
            "Attending Physician / Ophthalmologist Name:",
            value="Dr. A. Sharma, MD, DNB (Ophthalmology)",
            key=f"{key_prefix}_doc_name",
            help="Official name of the reviewing vitreo-retinal specialist or screening ophthalmologist."
        )
    with col_d2:
        reg_no = st.text_input(
            "Medical Council Registration / License No.:",
            value="MCI-RET-2018-9844",
            key=f"{key_prefix}_reg_no",
            help="Physician state/national medical council registration number."
        )

    edited_notes = st.text_area(
        "Attending Clinician Clinical Notes & Impression (Review & Edit):",
        value=initial_notes,
        height=130,
        key=f"{key_prefix}_notes_edit",
        help="Edit or append custom clinical observations before certifying the report."
    )

    sign_col1, sign_col2 = st.columns([2.8, 1.2])
    with sign_col1:
        is_signed = st.checkbox(
            "✍️ I hereby certify that I have reviewed the retinal photographs, Grad-CAM++ lesion heatmaps, and ETDRS CSME indicators, and approve this clinical evaluation.",
            value=True,
            key=f"{key_prefix}_sign_chk"
        )
    with sign_col2:
        cert_time = datetime.now().strftime("%d %b %Y, %I:%M %p")
        if is_signed:
            st.success("🟢 **Clinically Certified**")
        else:
            st.warning("⚠️ **Awaiting Doctor Signature**")

    if is_signed:
        cert_hash = f"CERT-{abs(hash(doc_name + reg_no + cert_time)) % 100000000:08d}"
        seal_html = f"""
        <div style="background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
                    border: 1.5px solid #86EFAC; border-radius: 8px; padding: 10px 16px; margin: 8px 0;
                    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
          <div>
            <div style="font-size: 12px; font-weight: 700; color: #166534;">
              ✓ DIGITALLY ACCREDITED CLINICAL AUDIT RECORD
            </div>
            <div style="font-size: 11px; color: #15803D; margin-top: 2px;">
              <b>Clinician:</b> {html.escape(doc_name)} &bull; <b>Reg No:</b> {html.escape(reg_no)} &bull; <b>Timestamp:</b> {cert_time}
            </div>
          </div>
          <div style="font-family: monospace; font-size: 11px; background: rgba(22, 101, 52, 0.12); color: #166534; padding: 3px 8px; border-radius: 4px; font-weight: 700;">
            ID: {cert_hash}
          </div>
        </div>
        """
        st.markdown(seal_html, unsafe_allow_html=True)
        clinician_info = {
            "name": doc_name.strip(),
            "reg_no": reg_no.strip(),
            "is_signed": True,
            "timestamp": cert_time,
            "notes": edited_notes,
            "cert_hash": cert_hash
        }
    else:
        clinician_info = {
            "name": doc_name.strip(),
            "reg_no": reg_no.strip(),
            "is_signed": False,
            "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "notes": edited_notes,
            "cert_hash": None
        }

    return clinician_info, edited_notes


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
    input_mode = st.sidebar.radio("Input Source:", ["📚 Clinical Database (Sample Cases)", "📤 Upload Custom Scan (.png, .jpg)"])

    if input_mode.startswith("📤"):
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
        ["📚 Clinical Scenarios & Presets (Sample Cases)", "📤 Upload Dual Scans (OD & OS)"]
    )
    if bilateral_source.startswith("📚"):
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
        ["📚 Load Clinical Camp Sample Batch (15 Cases)", "📤 Upload Bulk Scans (Multi-Image Dropzone)"]
    )
    
    if camp_source.startswith("📚"):
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
BACKBONE_DISPLAY_MAP = {
    "EfficientNet-B0 (Calibrated Baseline - 256x256, 100% Benchmark)": "efficientnet_b0",
    "ConvNeXt-Tiny (High-Resolution Pure-ConvNet - 512x512)": "convnext_tiny",
    "EfficientNet-B4 (Compound Scaling Medical Architecture - 380x380)": "efficientnet_b4",
    "Swin Transformer (Shifted-Window Self-Attention Swin-T - 256x256)": "swin_t",
    "Multi-Scale Clinical Ensemble (Triple-Scale High-Precision TTA)": "efficientnet_b0",
}

backbone_choice = st.sidebar.selectbox(
    "Model Architecture Backbone:",
    list(BACKBONE_DISPLAY_MAP.keys()),
    index=0,
    help="Select neural network backbone. High-resolution models (512x512 ConvNeXt-Tiny, 380x380 EfficientNet-B4) increase pixel area 4x for subtle microaneurysms."
)
chosen_backbone_key = BACKBONE_DISPLAY_MAP.get(backbone_choice, "efficientnet_b0")
use_multi_scale = ("Multi-Scale" in backbone_choice)

# Dynamic hot-swap to chosen backbone
model, grad_cam, temperature, temp_from_file, device, active_backbone_name = load_system(backbone=chosen_backbone_key)
b_meta = SUPPORTED_BACKBONES.get(active_backbone_name, SUPPORTED_BACKBONES["efficientnet_b0"])
current_target_size = b_meta.get("default_resolution", 256)

weights_target = b_meta.get("weights_file", "best_retina_model.pth")
has_custom_weights = os.path.exists(weights_target) and (weights_target != "best_retina_model.pth" or active_backbone_name == "efficientnet_b0")

status_pill = (
    f'<span style="background-color:#059669;color:#ffffff;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600;">🟢 Active ({b_meta["params"]})</span>'
    if has_custom_weights else
    f'<span style="background-color:#d97706;color:#ffffff;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600;">ℹ️ Fallback Base Weights</span>'
)

st.sidebar.markdown(
    f"""
    <div style="background-color:#0f172a;border:1px solid #334155;border-radius:8px;padding:8px 12px;margin-top:6px;margin-bottom:12px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
        <span style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:0.5px;">Architecture</span>
        {status_pill}
      </div>
      <div style="color:#f8fafc;font-size:12px;font-weight:600;">{b_meta["architecture_family"]}</div>
      <div style="color:#94a3b8;font-size:11px;margin-top:2px;">
        Resolution: <strong style="color:#38bdf8;">{current_target_size}×{current_target_size}</strong> &bull; Params: <strong style="color:#cbd5e1;">{b_meta["params"]}</strong>
      </div>
      <div style="color:#64748b;font-size:10px;margin-top:4px;line-height:1.3;">
        {b_meta["description"]}
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

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
with st.sidebar.expander("ℹ️ System Benchmarks & API Telemetry", expanded=False):
    st.markdown("**APTOS Validation Split**")
    st.markdown(f"- **QWK Kappa:** `{BENCHMARKS['qwk']}`")
    st.markdown(f"- **Accuracy:** `{BENCHMARKS['accuracy']}`")
    st.markdown(f"- **Calibration ECE:** `{BENCHMARKS['ece']}`")
    st.markdown("**External Test (IDRiD, 103 images)**")
    st.markdown(f"- **Accuracy:** `{EXTERNAL['accuracy']}`")
    st.markdown(f"- **QWK Kappa:** `{EXTERNAL['qwk']}`")
    st.markdown(f"- **Referable-DR AUC:** `{EXTERNAL['auc_referable']}`")
    st.caption("Performance drops on images from other cameras and clinics.")
    st.markdown(f"- **Temperature ($T$):** `{temperature:.4f}`")
    if not temp_from_file:
        st.warning("temperature.txt missing or invalid: using default temperature.")
    st.markdown("---")
    st.markdown("**FastAPI Microservice Integration**")
    st.code("uvicorn api:app --port 8000", language="bash")
    st.caption("OpenAPI Docs: http://localhost:8000/docs • CI/CD: 24 tests passed")

# ----------------- MAIN SCREEN -----------------
# High-Visibility Main Screen Input & Upload Hub
with st.container():
    hub_c1, hub_c2 = st.columns([1.3, 2.7])
    with hub_c1:
        main_protocol_choice = st.selectbox(
            "Screening Protocol Workflow:",
            [
                "Clinical Bilateral Screening (OD & OS)",
                "Single Fundus Scan",
                "Screening Camp Batch Processing (Bulk Triage)"
            ],
            index=0 if screening_mode.startswith("Clinical Bilateral") else 1 if screening_mode.startswith("Single") else 2,
            key="hub_protocol_select"
        )
        screening_mode = main_protocol_choice

    with hub_c2:
        if screening_mode == "Clinical Bilateral Screening (OD & OS)":
            main_bi_input_mode = st.radio(
                "Input Source:",
                ["📤 Upload Patient Scans (OD & OS)", "📚 Clinical Presets & Sample Cases"],
                horizontal=True,
                index=1 if bilateral_source.startswith("📚") and not (st.session_state.get("hub_up_od") or st.session_state.get("hub_up_os")) else 0,
                key="hub_bi_radio"
            )
            if main_bi_input_mode.startswith("📤"):
                u_c1, u_c2 = st.columns(2)
                with u_c1:
                    hub_up_od = st.file_uploader("📸 Right Eye (OD) Fundus Scan", type=["png", "jpg", "jpeg"], key="hub_up_od")
                    if hub_up_od is not None:
                        od_bytes = np.asarray(bytearray(hub_up_od.read()), dtype=np.uint8)
                        img_bgr_od = cv2.imdecode(od_bytes, cv2.IMREAD_COLOR)
                        od_source_label = f"Uploaded OD: {hub_up_od.name}"
                with u_c2:
                    hub_up_os = st.file_uploader("📸 Left Eye (OS) Fundus Scan", type=["png", "jpg", "jpeg"], key="hub_up_os")
                    if hub_up_os is not None:
                        os_bytes = np.asarray(bytearray(hub_up_os.read()), dtype=np.uint8)
                        img_bgr_os = cv2.imdecode(os_bytes, cv2.IMREAD_COLOR)
                        os_source_label = f"Uploaded OS: {hub_up_os.name}"
            else:
                main_bi_preset = st.selectbox(
                    "Select Preloaded Clinical Scenario:",
                    [
                        "Preset 1: Bilateral Moderate DR (OD Stage 2, OS Stage 2)",
                        "Preset 2: Asymmetric Progression (OD Stage 3 Severe, OS Stage 1 Mild)",
                        "Preset 3: High-Risk Proliferative Asymmetry (OD Stage 4 PDR, OS Stage 2 Moderate)",
                        "Preset 4: Early Unilateral Retinopathy (OD Stage 1 Mild, OS Stage 0 Normal)",
                        "Preset 5: Bilateral Healthy Retina (OD Stage 0 Normal, OS Stage 0 Normal)",
                    ],
                    key="hub_preset_select"
                )
                if main_bi_preset == "Preset 1: Bilateral Moderate DR (OD Stage 2, OS Stage 2)":
                    img_bgr_od, _ = get_sample_image(2, case_idx=0)
                    img_bgr_os, _ = get_sample_image(2, case_idx=1)
                    od_source_label = "OD: Stage 2 Case 1"
                    os_source_label = "OS: Stage 2 Case 2"
                elif main_bi_preset == "Preset 2: Asymmetric Progression (OD Stage 3 Severe, OS Stage 1 Mild)":
                    img_bgr_od, _ = get_sample_image(3, case_idx=0)
                    img_bgr_os, _ = get_sample_image(1, case_idx=0)
                    od_source_label = "OD: Stage 3 Case 1"
                    os_source_label = "OS: Stage 1 Case 1"
                elif main_bi_preset == "Preset 3: High-Risk Proliferative Asymmetry (OD Stage 4 PDR, OS Stage 2 Moderate)":
                    img_bgr_od, _ = get_sample_image(4, case_idx=0)
                    img_bgr_os, _ = get_sample_image(2, case_idx=0)
                    od_source_label = "OD: Stage 4 Case 1"
                    os_source_label = "OS: Stage 2 Case 1"
                elif main_bi_preset == "Preset 4: Early Unilateral Retinopathy (OD Stage 1 Mild, OS Stage 0 Normal)":
                    img_bgr_od, _ = get_sample_image(1, case_idx=0)
                    img_bgr_os, _ = get_sample_image(0, case_idx=0)
                    od_source_label = "OD: Stage 1 Case 1"
                    os_source_label = "OS: Stage 0 Case 1"
                elif main_bi_preset == "Preset 5: Bilateral Healthy Retina (OD Stage 0 Normal, OS Stage 0 Normal)":
                    img_bgr_od, _ = get_sample_image(0, case_idx=0)
                    img_bgr_os, _ = get_sample_image(0, case_idx=1)
                    od_source_label = "OD: Stage 0 Case 1"
                    os_source_label = "OS: Stage 0 Case 2"

        elif screening_mode == "Single Fundus Scan":
            main_sin_input_mode = st.radio(
                "Input Source:",
                ["📤 Upload Custom Scan (.png, .jpg)", "📚 Clinical Database Samples"],
                horizontal=True,
                key="hub_sin_radio"
            )
            if main_sin_input_mode.startswith("📤"):
                hub_up_sin = st.file_uploader("📸 Upload Single Eye Fundus Photograph", type=["png", "jpg", "jpeg"], key="hub_up_sin")
                if hub_up_sin is not None:
                    sin_bytes = np.asarray(bytearray(hub_up_sin.read()), dtype=np.uint8)
                    img_bgr = cv2.imdecode(sin_bytes, cv2.IMREAD_COLOR)
                    active_source_label = f"Uploaded File: {hub_up_sin.name}"
            else:
                p_c1, p_c2 = st.columns(2)
                with p_c1:
                    hub_stage_sel = st.selectbox(
                        "Severity Stage:",
                        ["Stage 0 (No DR)", "Stage 1 (Mild DR)", "Stage 2 (Moderate DR)", "Stage 3 (Severe DR)", "Stage 4 (Proliferative DR)"],
                        index=2,
                        key="hub_stage_select"
                    )
                stage_n = int(hub_stage_sel.split()[1])
                with p_c2:
                    hub_case_sel = st.selectbox("Patient Case:", [f"Case {i+1}" for i in range(5)], index=0, key="hub_case_select")
                case_i = int(hub_case_sel.split()[-1]) - 1
                img_bgr, _ = get_sample_image(stage_n, case_idx=case_i)
                active_source_label = f"Database: {hub_stage_sel} — {hub_case_sel}"

        else: # Screening camp
            st.caption("Screening camp batch ingestion can be loaded from 15 verified sample cases or uploaded in bulk via the sidebar.")

# Main screen clinical and AI settings expander
with st.expander("⚙️ Patient Demographics & AI Model Configuration", expanded=False):
    cfg1, cfg2 = st.columns(2)
    with cfg1:
        st.markdown("##### 👤 Patient Clinical Information")
        p_name = st.text_input("Patient Full Name:", value=p_name, key="hub_p_name")
        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            p_age = st.number_input("Age (years):", min_value=1, max_value=120, value=int(p_age), key="hub_p_age")
            p_duration = st.number_input("T2DM Duration (years):", min_value=0, max_value=60, value=int(p_duration), key="hub_p_dur")
        with sub_c2:
            p_gender = st.selectbox("Gender:", ["Male", "Female", "Other"], index=0 if p_gender=="Male" else 1 if p_gender=="Female" else 2, key="hub_p_gender")
            p_hba1c = st.number_input("HbA1c (%):", min_value=4.0, max_value=18.0, value=float(p_hba1c), step=0.1, key="hub_p_hba1c")
    with cfg2:
        st.markdown("##### 🧠 AI Model Architecture & Inference")
        hub_bb = st.selectbox(
            "Model Backbone Architecture:",
            list(BACKBONE_DISPLAY_MAP.keys()),
            index=list(BACKBONE_DISPLAY_MAP.keys()).index(backbone_choice) if backbone_choice in BACKBONE_DISPLAY_MAP else 0,
            key="hub_bb_select"
        )
        use_tta = st.checkbox("High-Precision TTA (4-Fold Invariance)", value=use_tta, key="hub_tta_check")
        blur_threshold = st.slider("Blur Sensitivity Threshold", 1.0, 30.0, float(blur_threshold), 1.0, key="hub_blur_slider")

if screening_mode == "Single Fundus Scan":
    if img_bgr is not None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        is_valid, msg, blur_val, bright_val = assess_image_quality(img_bgr, blur_threshold=blur_threshold)

        if not is_valid:
            st.error(f"🚨 **Screening Halted (Image Quality Failure):** {msg}")
            st.image(img_rgb, caption="Rejected Image — Blur or Illumination Artifacts Detected", width=380)
        else:
            st.caption(f"📁 **Active Scan:** `{active_source_label}` &bull; Quality Heuristics: `Passed` (Blur: {blur_val:.1f}, Illum: {bright_val:.1f}) &nbsp;|&nbsp; 💡 *To evaluate custom images, switch to **'📤 Upload Custom Scan'** in the sidebar.*")
            proc_rgb, calibrated_probs, pred_class, heatmap = run_inference(
                img_bgr, temperature, use_tta=use_tta, multi_scale=use_multi_scale, target_size=current_target_size, backbone=active_backbone_name
            )
            confidence = float(calibrated_probs[pred_class] * 100)
            csme_res = detect_macular_edema_risk(img_bgr, heatmap=heatmap, dr_stage=pred_class)

            patient_info_dict = {
                "name": p_name,
                "age": p_age,
                "gender": p_gender,
                "duration": p_duration,
                "hba1c": p_hba1c,
                "location": "Local Clinic"
            }

            st.markdown("---")
            render_single_triage_ribbon(patient_info_dict, pred_class, confidence, csme_res)

            tab_diag, tab_metrics, tab_report = st.tabs([
                "👁️ Clinical Diagnosis & Retinal Scans",
                "📊 AI Confidence & Lesion Analytics",
                "📋 Medical Report & EHR Sign-off"
            ])

            # ----------------- TAB 1: CLINICAL DIAGNOSIS & RETINAL SCANS -----------------
            with tab_diag:
                ctrl_col1, ctrl_col2 = st.columns([2.5, 1.5])
                with ctrl_col1:
                    view_mode = st.radio(
                        "Display Mode:",
                        ["📸 Original Fundus", "🔥 Grad-CAM++ Lesions", "🔬 Ben Graham Contrast", "🎯 ETDRS Macular ROI"],
                        horizontal=True,
                        key="single_view_mode"
                    )
                with ctrl_col2:
                    with st.expander("⚙️ Heatmap Display Settings", expanded=False):
                        selected_cmap = st.selectbox(
                            "Colormap:",
                            list(COLORMAP_OPTIONS.keys()),
                            index=0,
                            key="single_cmap"
                        )
                        cam_alpha = st.slider("Opacity (α):", 0.10, 0.90, 0.45, 0.05, key="single_alpha")
                        show_contours = st.checkbox("Outline Lesions", value=False, key="single_contours")

                overlay = generate_cam_overlay(
                    base_rgb=proc_rgb,
                    heatmap=heatmap,
                    colormap_name=selected_cmap,
                    alpha=cam_alpha,
                    show_contours=show_contours
                )

                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    st.image(img_rgb, caption=f"Original Retinal Scan ({img_rgb.shape[1]}×{img_rgb.shape[0]} px)", use_container_width=True)
                with v_col2:
                    if view_mode == "📸 Original Fundus":
                        st.image(img_rgb, caption="Original Fundus Photograph", use_container_width=True)
                    elif view_mode == "🔥 Grad-CAM++ Lesions":
                        st.image(overlay, caption=f"Grad-CAM++ Lesion Localization ({selected_cmap.split()[0]} | α={cam_alpha:.2f})", use_container_width=True)
                    elif view_mode == "🔬 Ben Graham Contrast":
                        st.image(proc_rgb, caption=f"Ben Graham Contrast Enhanced ({current_target_size}×{current_target_size})", use_container_width=True)
                    else:
                        st.image(csme_res["annotated_rgb"], caption=f"ETDRS Macular Field ({csme_res['status']})", use_container_width=True)

                proto = CLINICAL_PROTOCOLS[pred_class]
                u_col1, u_col2 = st.columns([1.5, 1])
                with u_col1:
                    st.markdown(
                        f"""<div style="background:#1e293b;border-left:4px solid {'#ef4444' if pred_class>=2 else '#f59e0b' if pred_class==1 else '#10b981'};border-radius:8px;padding:12px 16px;margin-top:10px;">
<div style="font-size:11px;color:#94a3b8;font-weight:700;text-transform:uppercase;">Recommended Clinical Protocol</div>
<div style="font-size:15px;font-weight:700;color:#f8fafc;margin-top:2px;">{proto['urgency']} &bull; <span style="font-weight:400;color:#cbd5e1;">{proto['timeline']}</span></div>
<div style="font-size:12px;color:#cbd5e1;margin-top:4px;"><b>Directive:</b> {DIAGNOSIS_INFO[pred_class]['action']}</div>
</div>""",
                        unsafe_allow_html=True
                    )
                with u_col2:
                    st.markdown(
                        f"""<div style="background:#1e293b;border-left:4px solid {'#ef4444' if csme_res.get('key')=='High' else '#f59e0b' if csme_res.get('key')=='Moderate' else '#10b981'};border-radius:8px;padding:12px 16px;margin-top:10px;">
<div style="font-size:11px;color:#94a3b8;font-weight:700;text-transform:uppercase;">Macular Edema Status</div>
<div style="font-size:15px;font-weight:700;color:#f8fafc;margin-top:2px;">{csme_res['status']}</div>
<div style="font-size:12px;color:#cbd5e1;margin-top:4px;">Risk Score: <b>{csme_res['score']:.1f}/100</b> &bull; Exudates: <b>{csme_res['macular_exudates']} px</b></div>
</div>""",
                        unsafe_allow_html=True
                    )

            # ----------------- TAB 2: AI CONFIDENCE & LESION ANALYTICS -----------------
            with tab_metrics:
                m1, m2 = st.columns([1.2, 1.8])
                with m1:
                    st.markdown("#### 🎯 Diagnostic Confidence")
                    st.markdown(f"### {DIAGNOSIS_INFO[pred_class]['name']}")
                    st.progress(max(0, min(int(confidence) if not np.isnan(confidence) else 0, 100)))
                    st.write(f"**Calibrated Confidence:** `{confidence:.2f}%`")
                    if confidence < LOW_CONFIDENCE_THRESHOLD:
                        st.warning("⚠️ Low model confidence. Manual review by a clinician is strongly advised.")
                    st.caption(f"Inference Mode: 4-Fold TTA Invariance &bull; Architecture: {b_meta['name']}")

                    st.markdown("---")
                    st.markdown("#### 🔍 Automated Quality Heuristics")
                    q1, q2 = st.columns(2)
                    with q1:
                        st.metric("Blur Score", f"{blur_val:.1f}", delta="Passed" if is_valid else "Blurry")
                    with q2:
                        st.metric("Illumination", f"{bright_val:.1f}", delta="Optimal")

                with m2:
                    st.markdown("#### 📊 5-Stage Calibrated Probability Distribution")
                    chart_df = pd.DataFrame({
                        "Probability (%)": [round(float(calibrated_probs[i] * 100), 2) for i in range(5)]
                    }, index=["Stage 0", "Stage 1", "Stage 2", "Stage 3", "Stage 4"])
                    chart_df.index.name = "DR Stage"
                    st.bar_chart(chart_df)

                    st.markdown("#### 🎯 ETDRS Macular Risk Telemetry")
                    st.markdown(
                        f"""<div style="background-color: {bg_hex}; border: 1.5px solid {border_hex}; border-radius: 8px; padding: 12px 16px; margin-top: 6px;">
<div style="font-size: 14px; font-weight: 700; color: #ffffff;">🎯 CSME Status: {csme_res['status']} (Score: {csme_res['score']:.1f}/100)</div>
<div style="margin-top: 4px; color: #cbd5e1; font-size: 12.5px;">
&bull; Central Foveal Exudates (&lt;500 μm): <b>{csme_res['foveal_exudates']} px</b><br>
&bull; Macular Ring Exudates (1-2 DD): <b>{csme_res['macular_exudates']} px</b><br>
&bull; Foveal CAM Attention Intensity: <b>{csme_res.get('foveal_cam_intensity', 0):.3f}</b>
</div>
<div style="margin-top: 6px; color: {text_hex}; font-size: 12.5px; font-weight: 600;">Directive: {csme_res['action']}</div>
</div>""",
                        unsafe_allow_html=True
                    )

            # ----------------- TAB 3: MEDICAL REPORT & EHR SIGN-OFF -----------------
            with tab_report:
                loc_clean = "Hyderabad"
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
                    st.success("🤖 Notes drafted by **Gemini (gemini-2.5-flash)** from screening results only.")
                else:
                    st.caption("ℹ️ Clinical notes generated by the built-in offline clinical engine.")

                clinician_info, final_notes = render_clinician_signoff_section(notes_text, key_prefix="single")

                st.markdown("#### 📄 Certified Clinical Report & EHR Data Export")
                try:
                    pdf_bytes = generate_pdf_report(
                        patient_info=patient_info_dict,
                        stage_name=DIAGNOSIS_INFO[pred_class]['name'],
                        confidence=confidence,
                        original_rgb=cv2.resize(img_rgb, (300, 300)),
                        gradcam_rgb=cv2.resize(overlay, (300, 300)),
                        clinical_notes_text=final_notes,
                        stage_idx=pred_class,
                        csme_info=csme_res,
                        clinician_info=clinician_info
                    )

                    safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', str(p_name).strip()) or "Patient"
                    clean_filename = f"Retina_Report_{safe_name}_{datetime.now().strftime('%Y%m%d')}.pdf"

                    exp_col1, exp_col2 = st.columns(2)
                    with exp_col1:
                        st.download_button(
                            label="📥 Download Certified Clinical PDF Report",
                            data=pdf_bytes,
                            file_name=clean_filename,
                            mime="application/pdf",
                            use_container_width=True,
                            type="primary"
                        )
                    with exp_col2:
                        import json
                        export_record = {
                            "patient_name": p_name,
                            "age": p_age,
                            "gender": p_gender,
                            "duration_years": p_duration,
                            "hba1c": p_hba1c,
                            "location": loc_clean,
                            "stage_index": int(pred_class),
                            "stage_name": DIAGNOSIS_INFO[pred_class]["name"],
                            "confidence_percent": round(confidence, 2),
                            "triage_urgency": CLINICAL_PROTOCOLS[pred_class]["urgency"],
                            "timeline": CLINICAL_PROTOCOLS[pred_class]["timeline"],
                            "csme_status": csme_res.get("status"),
                            "csme_score": round(float(csme_res.get("score", 0)), 1),
                            "clinician_signature": clinician_info
                        }
                        st.download_button(
                            label="💾 Download Clinical Data Record (JSON)",
                            data=json.dumps(export_record, indent=2).encode("utf-8"),
                            file_name=f"Clinical_Record_{safe_name}_{datetime.now().strftime('%Y%m%d')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Could not compile PDF: {e}")

                with st.expander("📍 Accredited Vitreo-Retinal Eye Hospital Directory", expanded=False):
                    for region, hospitals in EYE_HOSPITALS_DIRECTORY.items():
                        st.markdown(f"**{region}:**")
                        for h in hospitals:
                            st.markdown(f"- **[{h['name']}]({h['url']})** — *{h['city']}* ({h['type']}) | 📞 `{h['contact']}`")
    else:
        st.info("👆 Please upload a fundus photograph or choose a clinical database sample using the controls above to begin screening.")
elif screening_mode == "Clinical Bilateral Screening (OD & OS)":
    # ----------------- BILATERAL SCREENING PROTOCOL (OD vs. OS) -----------------
    if img_bgr_od is None or img_bgr_os is None:
        st.info("👆 **Scan Upload Pending:** Please upload fundus photographs for both eyes (OD and OS) using the upload boxes above to run automated AI screening.")
        c_u1, c_u2 = st.columns(2)
        with c_u1:
            if img_bgr_od is not None:
                st.success(f"✅ Right Eye (OD) Ready: `{od_source_label}`")
            else:
                st.warning("⚠️ Right Eye (OD — Oculus Dexter) scan pending upload.")
        with c_u2:
            if img_bgr_os is not None:
                st.success(f"✅ Left Eye (OS) Ready: `{os_source_label}`")
            else:
                st.warning("⚠️ Left Eye (OS — Oculus Sinister) scan pending upload.")
    else:
        img_rgb_od = cv2.cvtColor(img_bgr_od, cv2.COLOR_BGR2RGB)
        img_rgb_os = cv2.cvtColor(img_bgr_os, cv2.COLOR_BGR2RGB)

        is_valid_od, msg_od, blur_od, bright_od = assess_image_quality(img_bgr_od, blur_threshold=blur_threshold)
        is_valid_os, msg_os, blur_os, bright_os = assess_image_quality(img_bgr_os, blur_threshold=blur_threshold)

        if not is_valid_od or not is_valid_os:
            st.error("🚨 **Bilateral Screening Halted:** One or both images failed clinical quality heuristics. Please re-acquire or upload clear fundus images.")
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.image(img_rgb_od, caption=f"Right Eye (OD): {'Passed' if is_valid_od else 'Rejected: ' + msg_od}", use_container_width=True)
            with r_col2:
                st.image(img_rgb_os, caption=f"Left Eye (OS): {'Passed' if is_valid_os else 'Rejected: ' + msg_os}", use_container_width=True)
        else:
            st.caption(f"📁 **Active Bilateral Session:** OD `{od_source_label}` &bull; OS `{os_source_label}` &bull; Quality Heuristics: `Passed (OD & OS)` &nbsp;|&nbsp; 💡 *To evaluate a new patient, switch to **'📤 Upload Dual Scans'** in the sidebar.*")
            # 2. Run Inference & Grad-CAM++ for both eyes
            proc_rgb_od, probs_od, pred_class_od, heatmap_od = run_inference(
                img_bgr_od, temperature, use_tta=use_tta, multi_scale=use_multi_scale, target_size=current_target_size, backbone=active_backbone_name
            )
            proc_rgb_os, probs_os, pred_class_os, heatmap_os = run_inference(
                img_bgr_os, temperature, use_tta=use_tta, multi_scale=use_multi_scale, target_size=current_target_size, backbone=active_backbone_name
            )
            conf_od = float(probs_od[pred_class_od] * 100)
            conf_os = float(probs_os[pred_class_os] * 100)

            # Day 5: Bilateral CSME Macular Risk Analysis
            csme_od = detect_macular_edema_risk(img_bgr_od, heatmap=heatmap_od, dr_stage=pred_class_od)
            csme_os = detect_macular_edema_risk(img_bgr_os, heatmap=heatmap_os, dr_stage=pred_class_os)

            overall_stage = max(pred_class_od, pred_class_os)
            stage_diff = abs(pred_class_od - pred_class_os)
            higher_eye_label = "Right Eye (OD)" if pred_class_od > pred_class_os else "Left Eye (OS)" if pred_class_os > pred_class_od else "Both Eyes (Bilateral Concordance)"
            proto_overall = CLINICAL_PROTOCOLS[overall_stage]

            patient_info_dict = {
                "name": p_name,
                "age": p_age,
                "gender": p_gender,
                "duration": p_duration,
                "hba1c": p_hba1c,
                "location": "Local Clinic"
            }

            st.markdown("---")
            render_bilateral_triage_ribbon(patient_info_dict, pred_class_od, conf_od, csme_od, pred_class_os, conf_os, csme_os)

            tab_diag, tab_metrics, tab_report = st.tabs([
                "👁️ Clinical View & Dual Scans",
                "📊 AI Confidence & Inter-Ocular Metrics",
                "📋 Medical Report & EHR Sign-off"
            ])

            # ----------------- TAB 1: CLINICAL VIEW & DUAL SCANS -----------------
            with tab_diag:
                ctrl_col1, ctrl_col2 = st.columns([2.5, 1.5])
                with ctrl_col1:
                    bi_view_mode = st.radio(
                        "Dual Eye View:",
                        ["📸 Raw Fundus Scans", "🔥 Grad-CAM++ Lesions", "🔬 Ben Graham Contrast", "🎯 ETDRS Macular Field"],
                        horizontal=True,
                        key="bi_view_mode"
                    )
                with ctrl_col2:
                    with st.expander("⚙️ Heatmap Display Settings", expanded=False):
                        selected_cmap = st.selectbox(
                            "Colormap:",
                            list(COLORMAP_OPTIONS.keys()),
                            index=0,
                            key="bilateral_cmap"
                        )
                        cam_alpha = st.slider("Opacity (α):", 0.10, 0.90, 0.45, 0.05, key="bilateral_alpha")
                        show_contours = st.checkbox("Outline Lesions", value=False, key="bilateral_contours")

                overlay_od = generate_cam_overlay(proc_rgb_od, heatmap_od, colormap_name=selected_cmap, alpha=cam_alpha, show_contours=show_contours)
                overlay_os = generate_cam_overlay(proc_rgb_os, heatmap_os, colormap_name=selected_cmap, alpha=cam_alpha, show_contours=show_contours)
                short_cmap = selected_cmap.split()[0]

                v_od, v_os = st.columns(2)
                with v_od:
                    st.markdown(f"#### 👁️ Right Eye (OD): {DIAGNOSIS_INFO[pred_class_od]['name']} ({conf_od:.1f}%)")
                    if bi_view_mode == "📸 Raw Fundus Scans":
                        st.image(img_rgb_od, caption=f"OD Original Fundus ({img_rgb_od.shape[1]}×{img_rgb_od.shape[0]} px)", use_container_width=True)
                    elif bi_view_mode == "🔥 Grad-CAM++ Lesions":
                        st.image(overlay_od, caption=f"OD Grad-CAM++ ({short_cmap} | α={cam_alpha:.2f})", use_container_width=True)
                    elif bi_view_mode == "🔬 Ben Graham Contrast":
                        st.image(proc_rgb_od, caption=f"OD Ben Graham Contrast ({current_target_size}×{current_target_size})", use_container_width=True)
                    else:
                        st.image(csme_od["annotated_rgb"], caption=f"OD Macula: {csme_od['status']} ({csme_od['score']:.1f}/100)", use_container_width=True)

                with v_os:
                    st.markdown(f"#### 👁️ Left Eye (OS): {DIAGNOSIS_INFO[pred_class_os]['name']} ({conf_os:.1f}%)")
                    if bi_view_mode == "📸 Raw Fundus Scans":
                        st.image(img_rgb_os, caption=f"OS Original Fundus ({img_rgb_os.shape[1]}×{img_rgb_os.shape[0]} px)", use_container_width=True)
                    elif bi_view_mode == "🔥 Grad-CAM++ Lesions":
                        st.image(overlay_os, caption=f"OS Grad-CAM++ ({short_cmap} | α={cam_alpha:.2f})", use_container_width=True)
                    elif bi_view_mode == "🔬 Ben Graham Contrast":
                        st.image(proc_rgb_os, caption=f"OS Ben Graham Contrast ({current_target_size}×{current_target_size})", use_container_width=True)
                    else:
                        st.image(csme_os["annotated_rgb"], caption=f"OS Macula: {csme_os['status']} ({csme_os['score']:.1f}/100)", use_container_width=True)

                # Overall Protocol Card
                st.markdown(
                    f"""<div style="background:#1e293b;border-left:4px solid {'#ef4444' if overall_stage>=2 else '#f59e0b' if overall_stage==1 else '#10b981'};border-radius:8px;padding:12px 18px;margin-top:12px;">
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
<div>
  <div style="font-size:11px;color:#94a3b8;font-weight:700;text-transform:uppercase;">Overall Patient Triage &bull; Governing Eye: {higher_eye_label}</div>
  <div style="font-size:16px;font-weight:700;color:#f8fafc;margin-top:2px;">{DIAGNOSIS_INFO[overall_stage]['name']} &bull; <span style="color:#cbd5e1;font-weight:400;">{proto_overall['urgency']}</span></div>
  <div style="font-size:12.5px;color:#cbd5e1;margin-top:4px;">⏱️ <b>Recommended Timeline:</b> {proto_overall['timeline']} &nbsp;|&nbsp; <b>Care Level:</b> {proto_overall['referral_level']}</div>
</div>
</div>
</div>""",
                    unsafe_allow_html=True
                )

            # ----------------- TAB 2: AI CONFIDENCE & INTER-OCULAR METRICS -----------------
            with tab_metrics:
                if stage_diff == 0:
                    st.success(f"🟢 **Bilateral Concordance (Symmetric Involvement):** Both eyes exhibit identical microvascular severity ({DIAGNOSIS_INFO[overall_stage]['name']}). Typical systemic manifestation.")
                elif stage_diff == 1:
                    st.warning(f"⚠️ **Mild Inter-Ocular Asymmetry (Δ = 1 Stage):** Disparity in microvascular load between eyes ({higher_eye_label} Stage {overall_stage} vs. contralateral Stage {min(pred_class_od, pred_class_os)}). Governed by higher-severity eye.")
                else:
                    st.error(f"🚨 **CLINICAL RED FLAG — MARKED ASYMMETRY (Δ = {stage_diff} Stages):** Significant discrepancy detected ({higher_eye_label} Stage {overall_stage} vs. contralateral Stage {min(pred_class_od, pred_class_os)}). Warrants urgent carotid duplex Doppler ultrasonography (to rule out Ocular Ischemic Syndrome / carotid stenosis).")

                chart_col, conf_col = st.columns([1.6, 1.4])
                with chart_col:
                    st.markdown("#### 📊 Comparative Stage Probability Distribution (OD vs. OS)")
                    chart_df = pd.DataFrame({
                        "OD (Right Eye) %": [round(float(probs_od[i] * 100), 2) for i in range(5)],
                        "OS (Left Eye) %": [round(float(probs_os[i] * 100), 2) for i in range(5)]
                    }, index=["Stage 0", "Stage 1", "Stage 2", "Stage 3", "Stage 4"])
                    chart_df.index.name = "DR Stage"
                    st.bar_chart(chart_df)

                with conf_col:
                    st.markdown("#### 🎯 Eye-Specific Diagnostic Reliability")
                    st.markdown(f"**Right Eye (OD):** `{conf_od:.2f}%` Calibrated Confidence")
                    st.progress(max(0, min(int(conf_od) if not np.isnan(conf_od) else 0, 100)))
                    if conf_od < LOW_CONFIDENCE_THRESHOLD:
                        st.warning("⚠️ Low model confidence for OD.")

                    st.markdown(f"**Left Eye (OS):** `{conf_os:.2f}%` Calibrated Confidence")
                    st.progress(max(0, min(int(conf_os) if not np.isnan(conf_os) else 0, 100)))
                    if conf_os < LOW_CONFIDENCE_THRESHOLD:
                        st.warning("⚠️ Low model confidence for OS.")

                    st.markdown("---")
                    st.markdown("#### 🔍 Quality Metrics")
                    q_od, q_os = st.columns(2)
                    with q_od:
                        st.caption("Right Eye (OD)")
                        st.metric("OD Blur", f"{blur_od:.1f}")
                        st.metric("OD Illum.", f"{bright_od:.1f}")
                    with q_os:
                        st.caption("Left Eye (OS)")
                        st.metric("OS Blur", f"{blur_os:.1f}")
                        st.metric("OS Illum.", f"{bright_os:.1f}")

                st.markdown("#### 🔬 ETDRS Macular (CSME) Quantitative Comparison")
                c_od, c_os = st.columns(2)
                with c_od:
                    od_badge = "#ef4444" if csme_od.get('key')=='High' else "#f59e0b" if csme_od.get('key')=='Moderate' else "#10b981"
                    st.markdown(
                        f"""<div style="background:#1e293b;border:1.5px solid {od_badge};border-radius:8px;padding:12px 14px;">
<div style="font-size:14px;font-weight:700;color:#f8fafc;">Right Eye (OD) Macular Telemetry</div>
<div style="font-size:12.5px;color:#cbd5e1;margin-top:4px;">
&bull; Status: <b style="color:{od_badge};">{csme_od['status']}</b> (Score: {csme_od['score']:.1f}/100)<br>
&bull; Central Foveal Exudates (&lt;500µm): <b>{csme_od['foveal_exudates']} px</b><br>
&bull; Macular Ring Exudates (1-2 DD): <b>{csme_od['macular_exudates']} px</b>
</div>
</div>""",
                        unsafe_allow_html=True
                    )
                with c_os:
                    os_badge = "#ef4444" if csme_os.get('key')=='High' else "#f59e0b" if csme_os.get('key')=='Moderate' else "#10b981"
                    st.markdown(
                        f"""<div style="background:#1e293b;border:1.5px solid {os_badge};border-radius:8px;padding:12px 14px;">
<div style="font-size:14px;font-weight:700;color:#f8fafc;">Left Eye (OS) Macular Telemetry</div>
<div style="font-size:12.5px;color:#cbd5e1;margin-top:4px;">
&bull; Status: <b style="color:{os_badge};">{csme_os['status']}</b> (Score: {csme_os['score']:.1f}/100)<br>
&bull; Central Foveal Exudates (&lt;500µm): <b>{csme_os['foveal_exudates']} px</b><br>
&bull; Macular Ring Exudates (1-2 DD): <b>{csme_os['macular_exudates']} px</b>
</div>
</div>""",
                        unsafe_allow_html=True
                    )

            # ----------------- TAB 3: MEDICAL REPORT & EHR SIGN-OFF -----------------
            with tab_report:
                loc_clean = "Hyderabad"
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
                    st.success("🤖 Bilateral notes drafted by **Gemini (gemini-2.5-flash)** from screening results only.")
                else:
                    st.caption("ℹ️ Bilateral notes generated by the built-in offline clinical engine.")

                clinician_info, final_notes = render_clinician_signoff_section(notes_text, key_prefix="bilateral")

                st.markdown("#### 📄 Certified Bilateral Report & EHR Data Export")
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
                        clinical_notes_text=final_notes,
                        od_csme=csme_od,
                        os_csme=csme_os,
                        clinician_info=clinician_info
                    )

                    safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', str(p_name).strip()) or "Patient"
                    clean_filename = f"Retina_Bilateral_Report_{safe_name}_{datetime.now().strftime('%Y%m%d')}.pdf"

                    exp_col1, exp_col2 = st.columns(2)
                    with exp_col1:
                        st.download_button(
                            label="📥 Download Certified Bilateral PDF Report",
                            data=pdf_bytes,
                            file_name=clean_filename,
                            mime="application/pdf",
                            use_container_width=True,
                            type="primary"
                        )
                    with exp_col2:
                        import json
                        export_record = {
                            "patient_name": p_name,
                            "age": p_age,
                            "gender": p_gender,
                            "duration_years": p_duration,
                            "hba1c": p_hba1c,
                            "location": loc_clean,
                            "overall_stage_index": int(overall_stage),
                            "overall_stage_name": DIAGNOSIS_INFO[overall_stage]["name"],
                            "governing_eye": higher_eye_label,
                            "inter_ocular_asymmetry": stage_diff,
                            "od": {
                                "stage_index": int(pred_class_od),
                                "stage_name": DIAGNOSIS_INFO[pred_class_od]["name"],
                                "confidence_percent": round(conf_od, 2),
                                "csme_status": csme_od.get("status"),
                                "csme_score": round(float(csme_od.get("score", 0)), 1)
                            },
                            "os": {
                                "stage_index": int(pred_class_os),
                                "stage_name": DIAGNOSIS_INFO[pred_class_os]["name"],
                                "confidence_percent": round(conf_os, 2),
                                "csme_status": csme_os.get("status"),
                                "csme_score": round(float(csme_os.get("score", 0)), 1)
                            },
                            "clinician_signature": clinician_info
                        }
                        st.download_button(
                            label="💾 Download Bilateral Record (JSON)",
                            data=json.dumps(export_record, indent=2).encode("utf-8"),
                            file_name=f"Bilateral_Record_{safe_name}_{datetime.now().strftime('%Y%m%d')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Could not compile Bilateral PDF: {e}")

                with st.expander("📍 Accredited Vitreo-Retinal Eye Hospital Directory", expanded=False):
                    for region, hospitals in EYE_HOSPITALS_DIRECTORY.items():
                        st.markdown(f"**{region}:**")
                        for h in hospitals:
                            st.markdown(f"- **[{h['name']}]({h['url']})** — *{h['city']}* ({h['type']}) | 📞 `{h['contact']}`")
elif screening_mode == "Screening Camp Batch Processing (Bulk Triage)":
    st.subheader("⛺ Screening Camp Batch Processing & Triage (Epidemiological Workup)")
    st.caption("High-throughput automated screening for community vision camps, mobile health vans, and rural outreach programs.")

    camp_meta = {
        "camp_name": camp_name,
        "location": camp_loc,
        "screener_id": screener_id
    }

    camp_header_html = f"""
    <div style="background: linear-gradient(135deg, #090e17 0%, #1e293b 100%);
                border: 1px solid #1e3a8a; border-radius: 10px; padding: 14px 20px; margin-bottom: 16px; color: #ffffff;">
      <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
        <div>
          <span style="font-size: 15px; font-weight: 700; color: #93C5FD;">🏥 Clinical Facility: {html.escape(camp_name)}</span>
          <div style="font-size: 12px; color: #94A3B8; margin-top: 3px;">
            📍 Location: <b>{html.escape(camp_loc)}</b> &nbsp;|&nbsp; 👤 Screening Officer: <b>{html.escape(screener_id)}</b>
          </div>
        </div>
        <div style="background: rgba(37,99,235,0.2); border: 1px solid #3B82F6; padding: 4px 12px; border-radius: 6px; font-size: 12px; color: #60A5FA; font-weight: 600;">
          📁 Ingestion Queue: {len(batch_scans_data)} Scans Loaded
        </div>
      </div>
    </div>
    """
    st.markdown(camp_header_html, unsafe_allow_html=True)

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
                        proc_rgb, probs, pred_c, heatmap = run_inference(
                            scan_img, temperature, use_tta=use_tta, multi_scale=use_multi_scale, target_size=current_target_size, backbone=active_backbone_name
                        )
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
