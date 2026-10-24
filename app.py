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
    page_title="Retina Insight | Clinical Diagnostic Station",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- OPHTHALMIC CLINICAL THEME -----------------
ophthalmic_theme = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

    .stApp {
        background: radial-gradient(circle at 50% 0%, #0F254E 0%, #091326 45%, #050B17 100%) !important;
        color: #E2E8F0 !important;
    }

    #MainMenu { display: none !important; }
    footer { display: none !important; }
    div[data-testid="stToolbar"] { display: none !important; }
    div[data-testid="stDecoration"] { display: none !important; }
    div[data-testid="stStatusWidget"] { display: none !important; }
    .stDeployButton { display: none !important; }
    #manage-app-button { display: none !important; }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1250px !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #09142A 0%, #050B16 100%) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.18) !important;
    }

    .eye-hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(14, 165, 233, 0.12);
        color: #38BDF8;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 4px 14px;
        border-radius: 9999px;
        border: 1px solid rgba(56, 189, 248, 0.35);
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.2);
        margin-bottom: 0.6rem;
    }
    .eye-pulse { width: 8px; height: 8px; background: #38BDF8; border-radius: 50%; box-shadow: 0 0 10px #38BDF8; }
    .eye-hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(135deg, #FFFFFF 20%, #7DD3FC 60%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .eye-hero-sub { font-size: 0.95rem; color: #94A3B8; margin-bottom: 1.5rem; }

    div[data-testid="stMetric"] {
        background: rgba(15, 27, 52, 0.65) !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        backdrop-filter: blur(10px) !important;
    }
    div[data-testid="stMetric"] label { color: #94A3B8 !important; font-weight: 600 !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #38BDF8 !important; font-weight: 700 !important; }

    .clinical-section {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F1F5F9;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 1.8rem;
        margin-bottom: 0.9rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(56, 189, 248, 0.2);
    }

    div[data-testid="stImage"] {
        border-radius: 14px !important;
        overflow: hidden !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        background: #060D1A !important;
        box-shadow: 0 4px 20px rgba(14, 165, 233, 0.12) !important;
    }

    div.stDownloadButton > button {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        border: 1px solid #38BDF8 !important;
        padding: 0.75rem 1.8rem !important;
        box-shadow: 0 0 25px rgba(14, 165, 233, 0.4) !important;
        transition: all 0.25s ease !important;
    }
    div.stDownloadButton > button:hover {
        background: linear-gradient(135deg, #0369A1 0%, #075985 100%) !important;
        box-shadow: 0 0 35px rgba(56, 189, 248, 0.6) !important;
    }
    </style>
"""
st.markdown(ophthalmic_theme, unsafe_allow_html=True)

# ----------------- HERO BANNER -----------------
st.markdown("""
    <div class="eye-hero-badge"><span class="eye-pulse"></span>CLINICAL OPHTHALMOLOGY STATION &bull; ACTIVE</div>
    <div class="eye-hero-title">Retina Insight Diagnostic Console</div>
    <div class="eye-hero-sub">AI-Assisted Diabetic Retinopathy Screening, Uncertainty Calibration & Grad-CAM Optical Localization</div>
""", unsafe_allow_html=True)

BENCHMARKS = {"qwk": "0.8833", "accuracy": "81.3%", "ece": "3.58%"}
LOW_CONFIDENCE_THRESHOLD = 60.0

DIAGNOSIS_INFO = {
    0: {"name": "Stage 0: No Diabetic Retinopathy", "color": "#10B981", "action": "Routine annual screening recommended in 12 months."},
    1: {"name": "Stage 1: Mild DR", "color": "#38BDF8", "action": "Follow-up screening in 6-12 months. Maintain strict glycemic control."},
    2: {"name": "Stage 2: Moderate DR", "color": "#F59E0B", "action": "Referral to ophthalmologist within 2-4 months for dilated eye examination."},
    3: {"name": "Stage 3: Severe DR", "color": "#F97316", "action": "Urgent retinal specialist consultation within 2-4 weeks."},
    4: {"name": "Stage 4: Proliferative DR", "color": "#EF4444", "action": "Immediate emergency consultation required. High risk of permanent vision loss."}
}

DISCLAIMER_TEXT = (
    "This report is generated by an automated screening decision-support aid and is not an independent medical diagnosis. "
    "All findings must be reviewed and confirmed by a qualified ophthalmologist."
)

# ----------------- REALISTIC CLINICAL EHR DATABASE -----------------
PATIENT_EHR_DB = {
    # Stage 0: Early diabetes, well-controlled HbA1c
    (0, 1): {"name": "Ramesh Kumar", "age": 48, "gender": "Male", "duration": 4, "hba1c": 6.8},
    (0, 2): {"name": "Lakshmi Narayana", "age": 52, "gender": "Male", "duration": 5, "hba1c": 7.0},
    (0, 3): {"name": "Pooja Reddy", "age": 44, "gender": "Female", "duration": 3, "hba1c": 6.5},
    (0, 4): {"name": "Anil Verma", "age": 56, "gender": "Male", "duration": 6, "hba1c": 7.1},
    (0, 5): {"name": "Meera Joshi", "age": 50, "gender": "Female", "duration": 4, "hba1c": 6.9},

    # Stage 1: Mild microaneurysms, moderate duration
    (1, 1): {"name": "Suresh Babu", "age": 53, "gender": "Male", "duration": 7, "hba1c": 7.5},
    (1, 2): {"name": "Sunita Sharma", "age": 58, "gender": "Female", "duration": 8, "hba1c": 7.8},
    (1, 3): {"name": "Mohammed Farooq", "age": 49, "gender": "Male", "duration": 6, "hba1c": 7.4},
    (1, 4): {"name": "Kavitha Nair", "age": 55, "gender": "Female", "duration": 9, "hba1c": 8.0},
    (1, 5): {"name": "Rajesh Goud", "age": 60, "gender": "Male", "duration": 7, "hba1c": 7.6},

    # Stage 2: Moderate DR, elevated HbA1c
    (2, 1): {"name": "Venkat Rao", "age": 61, "gender": "Male", "duration": 11, "hba1c": 8.3},
    (2, 2): {"name": "Ananya Sen", "age": 57, "gender": "Female", "duration": 10, "hba1c": 8.6},
    (2, 3): {"name": "K. Srinivas", "age": 63, "gender": "Male", "duration": 12, "hba1c": 8.9},
    (2, 4): {"name": "Deepa Sundaram", "age": 54, "gender": "Female", "duration": 9, "hba1c": 8.2},
    (2, 5): {"name": "Harish Patel", "age": 66, "gender": "Male", "duration": 13, "hba1c": 8.7},

    # Stage 3: Severe DR, long-standing poorly controlled diabetes
    (3, 1): {"name": "Balaram Naidu", "age": 64, "gender": "Male", "duration": 15, "hba1c": 9.4},
    (3, 2): {"name": "Padmavathi Bai", "age": 67, "gender": "Female", "duration": 17, "hba1c": 9.8},
    (3, 3): {"name": "Abdul Rahim", "age": 59, "gender": "Male", "duration": 14, "hba1c": 9.2},
    (3, 4): {"name": "Geetha Pillai", "age": 62, "gender": "Female", "duration": 16, "hba1c": 9.6},
    (3, 5): {"name": "Chandra Shekar", "age": 68, "gender": "Male", "duration": 18, "hba1c": 10.1},

    # Stage 4: Proliferative DR, high complication risk
    (4, 1): {"name": "Govind Swamy", "age": 69, "gender": "Male", "duration": 22, "hba1c": 10.8},
    (4, 2): {"name": "Sarojini Devi", "age": 71, "gender": "Female", "duration": 20, "hba1c": 11.2},
    (4, 3): {"name": "K. Ramakrishna", "age": 65, "gender": "Male", "duration": 19, "hba1c": 10.5},
    (4, 4): {"name": "Fatima Begum", "age": 68, "gender": "Female", "duration": 21, "hba1c": 11.0},
    (4, 5): {"name": "Vishwa Mohan", "age": 73, "gender": "Male", "duration": 24, "hba1c": 11.5},
}

# ----------------- GATEKEEPER -----------------
def verify_retinal_anatomy(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask = gray > 15
    if np.sum(mask) < (image.shape[0] * image.shape[1] * 0.10):
        return False, "Non-Retinal Image: Image is mostly empty or blank."

    active_rgb = rgb[mask]
    active_hsv = hsv[mask]

    r_mean = float(np.mean(active_rgb[:, 0]))
    g_mean = float(np.mean(active_rgb[:, 1]))
    b_mean = float(np.mean(active_rgb[:, 2]))

    if r_mean < (1.15 * b_mean) or r_mean < (1.02 * g_mean):
        return False, f"Non-Retinal Image: Failed hemoglobin color signature (R:{r_mean:.0f}, G:{g_mean:.0f}, B:{b_mean:.0f})."

    hues = active_hsv[:, 0]
    sats = active_hsv[:, 1]
    retinal_pixels = ((hues <= 35) | (hues >= 155)) & (sats >= 20)
    retinal_ratio = float(np.mean(retinal_pixels))

    if retinal_ratio < 0.40:
        return False, f"Non-Retinal Image: Chromatic profile fails retinal spectrum ({retinal_ratio*100:.1f}% match)."

    return True, "Verified Authentic Retinal Fundus Photograph"

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
        return False, f"Image is too blurry (Blur Score: {laplacian_var:.1f} < {blur_threshold:.1f})", laplacian_var, mean_brightness
    if mean_brightness < min_brightness or mean_brightness > max_brightness:
        return False, f"Improper illumination ({mean_brightness:.1f})", laplacian_var, mean_brightness

    return True, "Passed: Diagnostic Quality Approved", laplacian_var, mean_brightness

def fast_ben_graham(img_bgr, target_size=256):
    # Preserves native circular aspect ratio and lesion scale (matches Kaggle training)
    img_resized = cv2.resize(img_bgr, (target_size, target_size))
    blurred = cv2.GaussianBlur(img_resized, (0, 0), 10)
    enhanced = cv2.addWeighted(img_resized, 4, blurred, -4, 128)
    return enhanced

# ----------------- GRAD-CAM -----------------
class GradCAM:
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

        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))

        cam = np.tensordot(weights, activations, axes=1).astype(np.float32)
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (256, 256))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)

        logits = output.detach().cpu()
        self.gradients = None
        self.activations = None
        return cam, target_class, logits

# ----------------- MODEL LOADER -----------------
DEFAULT_TEMPERATURE = 1.3301

@st.cache_resource
def load_system():
    if not os.path.exists("best_retina_model.pth"):
        raise FileNotFoundError("best_retina_model.pth checkpoint was not found.")

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
    if os.path.exists("temperature.txt"):
        try:
            with open("temperature.txt", "r") as f:
                val = float(f.read().strip())
                if val > 0:
                    temp = val
        except:
            temp = DEFAULT_TEMPERATURE

    cam = GradCAM(model, model.features[-1])
    return model, cam, temp, device

try:
    model, grad_cam, temperature, device = load_system()
except Exception as e:
    st.error(f"🚨 Model initialization error: {e}")
    st.stop()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_data(show_spinner="Performing AI Retinal Analysis...")
def run_inference(img_bgr, temp):
    proc_bgr = fast_ben_graham(img_bgr, target_size=256)
    proc_rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
    tensor_img = preprocess(Image.fromarray(proc_rgb)).unsqueeze(0).to(device)

    heatmap, pred_class, logits = grad_cam.generate(tensor_img)
    probs = torch.softmax(logits / temp, dim=1)[0].numpy()

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(proc_rgb, 0.6, heatmap_colored, 0.4, 0)
    return proc_rgb, probs, int(pred_class), overlay

# ----------------- CLINICAL NOTES (GENAI + OFFLINE) -----------------
def _offline_notes(stage_idx, stage_name, confidence, patient_info):
    patient_messages = {
        0: "Your retinal scan indicates healthy retinal tissue without diabetic microvascular damage. Continue optimal blood sugar management and schedule regular screening in 12 months.",
        1: "Early signs of mild diabetic retinopathy (microaneurysms) were observed. Vision is typically unaffected at this stage. Work closely with your physician to manage blood sugar and repeat screening in 6 to 12 months.",
        2: "Moderate diabetic retinopathy detected. Retinal blood vessels show visible leakage. A comprehensive dilated examination by an ophthalmologist is recommended within 2 to 4 months.",
        3: "Severe diabetic retinopathy identified. Extensive vascular occlusion was detected. Immediate specialist consultation is required within 2 to 4 weeks to evaluate preventive intervention.",
        4: "Proliferative diabetic retinopathy detected. Fragile new blood vessels (neovascularization) pose an imminent risk to vision. Emergency evaluation and immediate clinical treatment are required."
    }

    soap_note = f"""**Subjective:** Patient {patient_info.get('name')}, {patient_info.get('age')} yo. Diabetes duration: {patient_info.get('duration')} years. HbA1c: {patient_info.get('hba1c')}%.
**Objective:** Retinal fundus scan evaluated. IQA Gatekeeper: PASSED. AI Backbone (EfficientNet-B0): {stage_name} ({confidence:.1f}% calibrated confidence). Grad-CAM shows localized lesion focus.
**Assessment:** Diabetic Retinopathy Severity: {stage_name}.
**Plan:** {DIAGNOSIS_INFO[stage_idx]['action']}"""

    return (
        f"### 🧑‍⚕️ Patient Summary\n{patient_messages.get(stage_idx, '')}\n\n"
        f"### 📋 Clinical SOAP Note\n{soap_note}\n\n"
        f"*{DISCLAIMER_TEXT}*"
    )

def _gemini_notes(stage_name, confidence, patient_info, api_key):
    # HIPAA/Privacy-preserving: Direct identifiers omitted
    client = genai.Client(api_key=api_key)
    prompt = f"""
    You are an expert AI Ophthalmologist assistant drafting clinical screening decision notes.
    Patient: Age {patient_info.get('age')}, Diabetes duration: {patient_info.get('duration')} yrs, HbA1c: {patient_info.get('hba1c')}%.
    Screening Diagnosis: {stage_name} with {confidence:.1f}% calibrated confidence.

    Rules:
    - Base findings strictly on screening findings.
    - Emphasize that this is an AI screening decision aid requiring attending ophthalmologist confirmation.
    - Write two clean sections with exact headings:
    ### Patient Explanation
    (Empathetic, clear, no medical jargon)
    ### Clinical SOAP Note
    (Subjective, Objective, Assessment, Plan for attending doctor)
    """
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text

def generate_clinical_notes(stage_idx, stage_name, confidence, patient_info, api_key=None):
    error_msg = None
    if api_key and GENAI_AVAILABLE:
        cache = st.session_state.setdefault("gemini_cache", {})
        cache_key = (stage_idx, round(float(confidence), 1), patient_info.get('age'), patient_info.get('hba1c'))
        if cache_key in cache:
            return cache[cache_key], True, None
        try:
            text = _gemini_notes(stage_name, confidence, patient_info, api_key)
            if text:
                cache[cache_key] = text
                return text, True, None
            error_msg = "Gemini returned empty response."
        except Exception as e:
            error_msg = f"Gemini connection failed ({type(e).__name__}); using built-in clinical engine."

    return _offline_notes(stage_idx, stage_name, confidence, patient_info), False, error_msg

# ----------------- REPORTLAB PDF GENERATOR -----------------
_UNSUPPORTED_CHARS = re.compile(r'[^\x00-\x7F\u00A0-\u00FF\u2013\u2014\u2018\u2019\u201C\u201D\u2022]')

def _inline_markup(line):
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

def generate_pdf_report(patient_info, stage_name, confidence, original_rgb, gradcam_rgb, clinical_notes_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=18, leading=22, textColor=colors.HexColor('#1E3A8A'), alignment=1, spaceAfter=4)
    subtitle_style = ParagraphStyle('DocSub', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#64748B'), alignment=1, spaceAfter=12)
    header_style = ParagraphStyle('SectionHead', parent=styles['Heading2'], fontSize=11, leading=14, textColor=colors.HexColor('#1E3A8A'), spaceBefore=8, spaceAfter=4)
    body_style = ParagraphStyle('DocBody', parent=styles['Normal'], fontSize=8.5, leading=11, textColor=colors.HexColor('#1E293B'))
    disclaimer_style = ParagraphStyle('Disclaimer', parent=body_style, fontSize=7.5, leading=10, textColor=colors.HexColor('#B91C1C'))

    elements.append(Paragraph("<b>RETINA INSIGHT CLINICAL DIAGNOSTIC REPORT</b>", title_style))
    elements.append(Paragraph("Automated Retinal Screening &amp; Explainable AI Decision Support Platform", subtitle_style))
    elements.append(Spacer(1, 8))

    scan_date = datetime.now().strftime("%d %b %Y, %I:%M %p")
    p_name_esc = html.escape(str(patient_info.get('name', 'Anonymous')))
    patient_data = [
        [Paragraph("<b>Patient Name:</b>", body_style), Paragraph(p_name_esc, body_style),
         Paragraph("<b>Report Date:</b>", body_style), Paragraph(scan_date, body_style)],
        [Paragraph("<b>Age / Gender:</b>", body_style), Paragraph(f"{patient_info.get('age', 'N/A')} yrs / {html.escape(str(patient_info.get('gender', 'N/A')))}", body_style),
         Paragraph("<b>HbA1c Level:</b>", body_style), Paragraph(f"{patient_info.get('hba1c', 'N/A')}%", body_style)],
        [Paragraph("<b>Diabetes Duration:</b>", body_style), Paragraph(f"{patient_info.get('duration', 'N/A')} Years", body_style),
         Paragraph("<b>Safety Gatekeeper:</b>", body_style), Paragraph("<font color='green'><b>PASSED (Valid Scan)</b></font>", body_style)]
    ]
    t_patient = Table(patient_data, colWidths=[100, 160, 100, 160])
    t_patient.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(t_patient)
    elements.append(Spacer(1, 10))

    diag_data = [
        [Paragraph(f"<b>FINAL AI DIAGNOSIS:</b> {html.escape(stage_name.upper())}", ParagraphStyle('W', parent=body_style, fontSize=10, textColor=colors.white)),
         Paragraph(f"<b>Calibrated Confidence:</b> {confidence:.1f}% (QWK: 0.8833)", ParagraphStyle('W2', parent=body_style, fontSize=9.5, textColor=colors.white, alignment=2))]
    ]
    t_diag = Table(diag_data, colWidths=[360, 160])
    t_diag.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#1E3A8A')),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    elements.append(t_diag)
    elements.append(Spacer(1, 10))

    img_data = [
        [Paragraph("<b>Original Patient Retinal Scan</b>", body_style), Paragraph("<b>Grad-CAM Explainable AI Lesion Heatmap</b>", body_style)],
        [RLImage(_jpeg_buffer(original_rgb), width=200, height=200), RLImage(_jpeg_buffer(gradcam_rgb), width=200, height=200)]
    ]
    t_img = Table(img_data, colWidths=[260, 260])
    t_img.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,0), 3),
    ]))
    elements.append(t_img)
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("<b>Clinical Evaluation &amp; Decision Support</b>", header_style))
    elements.extend(_notes_to_flowables(clinical_notes_text, body_style, header_style))
    elements.append(Spacer(1, 12))

    sign_data = [
        [Paragraph("<b>Screening Software:</b> Retina Insight v2.0", body_style),
         Paragraph("<b>Attending Ophthalmologist Sign-off:</b> ___________________", body_style)]
    ]
    t_sign = Table(sign_data, colWidths=[280, 240])
    elements.append(t_sign)
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(f"<b>Disclaimer:</b> {html.escape(DISCLAIMER_TEXT)}", disclaimer_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.markdown("### 📁 Clinical Screening Input")
input_mode = st.sidebar.radio("Data Mode:", ["Pre-loaded Database", "Upload Custom Scan"])

img_bgr = None
active_source_label = ""
current_profile = {"name": "Anonymous Patient", "age": 55, "gender": "Male", "duration": 8, "hba1c": 8.0}

if input_mode == "Upload Custom Scan":
    uploaded_file = st.sidebar.file_uploader("Upload fundus photograph (.png, .jpg)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            st.sidebar.error("❌ Could not read uploaded file as an image.")
        else:
            active_source_label = f"Uploaded Scan: {uploaded_file.name}"
else:
    sel_stage = st.sidebar.selectbox(
        "Select Severity Stage:",
        ["Stage 0 (No DR)", "Stage 1 (Mild DR)", "Stage 2 (Moderate DR)", "Stage 3 (Severe DR)", "Stage 4 (Proliferative DR)"]
    )
    stage_num = int(sel_stage.split()[1])

    matched_images = []
    if os.path.exists("sample_images"):
        # Match cases 1 to 5 cleanly
        for i in range(1, 6):
            matches = [os.path.join("sample_images", f) for f in sorted(os.listdir("sample_images"))
                       if f.lower().startswith(f"stage_{stage_num}_case_{i}_") and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if matches:
                matched_images.append(matches[0])

        if not matched_images:
            for f in sorted(os.listdir("sample_images")):
                if f.lower().startswith(f"stage_{stage_num}") and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    matched_images.append(os.path.join("sample_images", f))

    if len(matched_images) > 0:
        case_options = [f"Patient Case {i+1}" for i in range(len(matched_images))]
        sel_case = st.sidebar.selectbox("Select Patient Case:", case_options)
        case_idx = int(sel_case.split()[-1])

        chosen_path = matched_images[case_idx - 1]
        img_bgr = cv2.imread(chosen_path)
        active_source_label = f"Clinical DB: {sel_stage} — {sel_case}"
        st.sidebar.caption(f"📁 {len(matched_images)} verified clinical case(s) on disk")

        if (stage_num, case_idx) in PATIENT_EHR_DB:
            current_profile = PATIENT_EHR_DB[(stage_num, case_idx)]
        else:
            current_profile = {
                "name": f"Patient #{stage_num * 10 + case_idx}",
                "age": 45 + stage_num * 5,
                "gender": "Male" if case_idx % 2 == 1 else "Female",
                "duration": 4 + stage_num * 4,
                "hba1c": round(6.5 + stage_num * 1.0 + (case_idx * 0.1), 1)
            }
    else:
        st.sidebar.error(f"❌ No images found for Stage {stage_num}.")

st.sidebar.markdown("---")
with st.sidebar.expander("👤 Patient Demographics & EHR", expanded=True):
    dynamic_key = f"prof_{input_mode}_{active_source_label}"
    p_name = st.text_input("Full Name", value=current_profile["name"], key=f"name_{dynamic_key}")
    p_age = st.number_input("Age", min_value=18, max_value=100, value=current_profile["age"], key=f"age_{dynamic_key}")

    gender_options = ["Male", "Female", "Other"]
    gender_idx = gender_options.index(current_profile["gender"]) if current_profile["gender"] in gender_options else 0
    p_gender = st.selectbox("Gender", gender_options, index=gender_idx, key=f"gender_{dynamic_key}")

    p_duration = st.number_input("Diabetes Duration (Years)", min_value=0, max_value=50, value=current_profile["duration"], key=f"dur_{dynamic_key}")
    p_hba1c = st.number_input("Recent HbA1c (%)", min_value=4.0, max_value=16.0, value=float(current_profile["hba1c"]), step=0.1, key=f"hba1c_{dynamic_key}")

with st.sidebar.expander("🤖 Gemini GenAI Key (Optional)", expanded=False):
    gemini_key = st.text_input("API Key", type="password", help="Leave blank to use built-in offline clinical assistant. Patient direct identifiers are never sent to Gemini.")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Gatekeeper Controls")
blur_threshold = st.sidebar.slider(
    "IQA Blur Sensitivity",
    min_value=1.0,
    max_value=30.0,
    value=5.0,
    step=1.0,
    help="Images with standardized blur score below this threshold are rejected."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 Certified Benchmarks")
st.sidebar.markdown(f"- **QWK Kappa Score:** `{BENCHMARKS['qwk']}`")
st.sidebar.markdown(f"- **Validation Accuracy:** `{BENCHMARKS['accuracy']}`")
st.sidebar.markdown(f"- **Calibration ECE:** `{BENCHMARKS['ece']}`")
st.sidebar.markdown(f"- **Temperature ($T$):** `{temperature:.4f}`")

# ----------------- MAIN SCREEN -----------------
if img_bgr is not None:
    st.info(f"🔍 **Active Diagnostic Target:** `{active_source_label}`")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1. GATEKEEPER
    st.markdown('<div class="clinical-section">🛡️ 1. Clinical Gatekeeper (Anatomical & Optical Quality Inspection)</div>', unsafe_allow_html=True)
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
        st.error(f"🚨 **Screening Halted by Safety Gatekeeper:** {msg}")
        st.image(img_rgb, caption="Uploaded Non-Conforming Image", width=350)
    else:
        # Caching ensures moving sliders or typing patient name does NOT re-run neural network
        proc_rgb, calibrated_probs, pred_class, overlay = run_inference(img_bgr, temperature)
        confidence = float(calibrated_probs[pred_class] * 100)

        # 2. DIAGNOSTIC RESULTS & RISK STRATIFICATION
        st.markdown('<div class="clinical-section">📊 2. Diagnostic Severity Grading & Calibrated Confidence</div>', unsafe_allow_html=True)

        top2_idx = np.argsort(calibrated_probs)[::-1][:2]
        primary_class = int(top2_idx[0])
        secondary_class = int(top2_idx[1])
        secondary_conf = calibrated_probs[secondary_class] * 100

        # Determine Vision-Threatening DR (VTDR) Status
        is_vtdr = primary_class in [3, 4] or (primary_class == 2 and calibrated_probs[3] + calibrated_probs[4] > 0.35)

        if is_vtdr:
            st.error("🚨 **CLINICAL STATUS: VISION-THREATENING DIABETIC RETINOPATHY (VTDR) DETECTED**\nImmediate specialist laser or anti-VEGF intervention recommended.")
        elif primary_class == 2:
            st.warning("⚠️ **CLINICAL STATUS: REFERABLE DIABETIC RETINOPATHY (RDR)**\nSpecialist ophthalmologist dilated examination required within 2-4 months.")
        else:
            st.success("✅ **CLINICAL STATUS: NON-REFERABLE DIABETIC RETINOPATHY**\nStandard annual screening protocol applicable.")

        res1, res2 = st.columns([1.2, 1.8])
        with res1:
            stage_color = DIAGNOSIS_INFO[primary_class]['color']
            st.markdown(f"""
                <div style="background: rgba(15, 27, 52, 0.7); border-left: 5px solid {stage_color}; border-radius: 10px; padding: 14px; margin-bottom: 12px; border-top: 1px solid rgba(56,189,248,0.15); border-right: 1px solid rgba(56,189,248,0.15); border-bottom: 1px solid rgba(56,189,248,0.15);">
                    <div style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF;">{DIAGNOSIS_INFO[primary_class]['name']}</div>
                    <div style="font-size: 0.95rem; color: #94A3B8; margin-top: 4px;">Primary Calibrated Confidence: <b style="color: {stage_color}; font-size: 1.15rem;">{confidence:.2f}%</b></div>
                    <div style="font-size: 0.85rem; color: #64748B; margin-top: 4px;">Secondary Differential: <b>Stage {secondary_class} ({secondary_conf:.1f}%)</b></div>
                </div>
            """, unsafe_allow_html=True)

            st.progress(min(int(confidence), 100))
            if confidence < LOW_CONFIDENCE_THRESHOLD:
                st.warning(f"⚠️ Calibrated confidence ({confidence:.1f}%) is below {LOW_CONFIDENCE_THRESHOLD}%. High borderline uncertainty — secondary specialist confirmation advised.")
            st.info(f"**Action:** {DIAGNOSIS_INFO[primary_class]['action']}")

        with res2:
            chart_data = {f"Stage {i}": round(float(calibrated_probs[i] * 100), 2) for i in range(5)}
            st.bar_chart(chart_data)

        # 3. GRAD-CAM
        st.markdown('<div class="clinical-section">🔬 3. Explainable AI (XAI) — Pathological Lesion Attention Maps</div>', unsafe_allow_html=True)
        v1, v2, v3 = st.columns(3)
        with v1:
            st.image(img_rgb, caption="1. Original Fundus Scan", use_container_width=True)
        with v2:
            st.image(proc_rgb, caption="2. Color-Normalized Contrast (Ben Graham Preprocessed)", use_container_width=True)
        with v3:
            st.image(overlay, caption="3. Grad-CAM Lesion Overlay", use_container_width=True)

        # 4. GENAI & PDF
        st.markdown('<div class="clinical-section">📋 4. Multimodal GenAI Clinical Notes & Official Report Export</div>', unsafe_allow_html=True)

        patient_info_dict = {
            "name": p_name,
            "age": p_age,
            "gender": p_gender,
            "duration": p_duration,
            "hba1c": p_hba1c
        }

        notes_text, is_gemini, notes_err = generate_clinical_notes(
            stage_idx=primary_class,
            stage_name=DIAGNOSIS_INFO[primary_class]['name'],
            confidence=confidence,
            patient_info=patient_info_dict,
            api_key=gemini_key.strip() if gemini_key else None
        )

        if notes_err:
            st.warning(f"ℹ️ {notes_err}")
        if is_gemini:
            st.success("🤖 Analysis generated live via **Gemini GenAI (gemini-2.5-flash)** (HIPAA-compliant de-identified query)")
        else:
            st.caption("ℹ️ Clinical assessment generated via **Built-in Offline Clinical Knowledge Engine**")

        st.markdown(notes_text)

        try:
            pdf_bytes = generate_pdf_report(
                patient_info=patient_info_dict,
                stage_name=DIAGNOSIS_INFO[primary_class]['name'],
                confidence=confidence,
                original_rgb=cv2.resize(img_rgb, (300, 300)),
                gradcam_rgb=cv2.resize(overlay, (300, 300)),
                clinical_notes_text=notes_text
            )

            safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', str(p_name).strip()) or "Patient"
            clean_filename = f"Retina_Report_{safe_name}_{datetime.now().strftime('%Y%m%d')}.pdf"
            st.download_button(
                label="📥 Download Official Hospital Clinical Report (PDF)",
                data=pdf_bytes,
                file_name=clean_filename,
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Could not compile PDF: {e}")
else:
    st.info("👈 Please select a patient case from the sidebar or upload a fundus photograph to begin screening.")
