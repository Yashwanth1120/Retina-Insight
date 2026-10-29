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

# ----------------- GRAD-CAM -----------------
class GradCAM:
    """Grad-CAM using a forward hook + tensor hook (avoids full_backward_hook in-place issues)."""

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

    cam = GradCAM(model, model.features[-1])
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
def run_inference(img_bgr, temp):
    """Preprocess, predict, and build the Grad-CAM overlay in a single forward/backward pass.
    Cached so typing in the patient form does not re-run the model."""
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
        0: "The automated screening model did not find signs of diabetic retinopathy in this image. Continue good blood sugar control and keep up regular eye screening (about every 12 months, or as your doctor advises).",
        1: "The automated screening model suggests mild diabetic retinopathy, the earliest stage. Vision is usually unaffected at this stage. Work with your doctor on blood sugar control and repeat screening in 6 to 12 months.",
        2: "The automated screening model suggests moderate diabetic retinopathy. Please arrange a dilated eye examination with an ophthalmologist within 2 to 4 months.",
        3: "The automated screening model suggests severe diabetic retinopathy. Please see a retina specialist within 2 to 4 weeks.",
        4: "The automated screening model suggests proliferative diabetic retinopathy, the most advanced stage. Please seek urgent evaluation by an eye specialist."
    }

    soap_note = f"""**Subjective:** {patient_info.get('age')} yo patient. Diabetes duration: {patient_info.get('duration')} years. HbA1c: {patient_info.get('hba1c')}%.
**Objective:** Fundus image passed automated quality heuristics. EfficientNet-B0 screening output: {stage_name} ({confidence:.1f}% calibrated confidence). The Grad-CAM map highlights regions that influenced the model; it has not been validated as lesion localization.
**Assessment:** AI-suggested diabetic retinopathy severity: {stage_name}. Requires clinician confirmation.
**Plan (suggested):** {DIAGNOSIS_INFO[stage_idx]['action']}"""

    return (
        f"### Patient Summary\n{patient_messages.get(stage_idx, '')}\n\n"
        f"### Clinical SOAP Note\n{soap_note}\n\n"
        f"*{DISCLAIMER_TEXT}*"
    )

def _gemini_notes(stage_name, confidence, patient_info, api_key):
    # No patient name or other direct identifiers are sent to the external API.
    client = genai.Client(api_key=api_key)
    prompt = f"""
You are an assistant helping draft notes for a diabetic retinopathy screening report.
Patient: {patient_info.get('age')} years old, diabetes duration {patient_info.get('duration')} years, HbA1c {patient_info.get('hba1c')}%.
Automated screening result: {stage_name} with {confidence:.1f}% calibrated confidence.

Rules:
- Base your text ONLY on the screening result above. Do not claim to have seen the image or to have detected specific lesions.
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

def generate_clinical_notes(stage_idx, stage_name, confidence, patient_info, api_key=None):
    """Returns (notes_text, used_gemini, error_message). Gemini results are cached per session
    so Streamlit reruns do not trigger repeated API calls."""
    error_msg = None
    if api_key and GENAI_AVAILABLE:
        cache = st.session_state.setdefault("gemini_cache", {})
        key = (stage_idx, round(float(confidence), 1), patient_info.get('age'),
               patient_info.get('duration'), patient_info.get('hba1c'))
        if key in cache:
            return cache[key], True, None
        try:
            text = _gemini_notes(stage_name, confidence, patient_info, api_key)
            if text:
                cache[key] = text
                return text, True, None
            error_msg = "Gemini returned an empty response."
        except Exception as e:
            error_msg = f"Gemini request failed ({type(e).__name__}); showing the offline summary instead."
    elif api_key and not GENAI_AVAILABLE:
        error_msg = "google-genai is not installed; showing the offline summary instead."

    return _offline_notes(stage_idx, stage_name, confidence, patient_info), False, error_msg

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

def generate_pdf_report(patient_info, stage_name, confidence, original_rgb, gradcam_rgb, clinical_notes_text, stage_idx=0):
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
    elements.append(Paragraph("Automated Retinal Screening &amp; Explainable AI Decision Support", subtitle_style))
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
         Paragraph("<b>Image Quality Check:</b>", body_style), Paragraph("Passed (automated heuristic)", body_style)]
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
    diag_data = [
        [Paragraph(f"<b>AI-SUGGESTED STAGE:</b> {html.escape(stage_name.upper())}", ParagraphStyle('W', parent=body_style, fontSize=10, textColor=colors.white)),
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
    proto = CLINICAL_PROTOCOLS.get(int(stage_idx if 'stage_idx' in locals() else 0), CLINICAL_PROTOCOLS[0])
    referral_rows = [
        [Paragraph("<b>Clinical Triage Urgency:</b>", body_style), Paragraph(f"<b>{html.escape(proto['urgency'])}</b>", body_style),
         Paragraph("<b>Recommended Timeline:</b>", body_style), Paragraph(f"<b>{html.escape(proto['timeline'])}</b>", body_style)],
        [Paragraph("<b>Referral Level:</b>", body_style), Paragraph(html.escape(proto['referral_level']), body_style),
         Paragraph("<b>Recommended Workup:</b>", body_style), Paragraph(html.escape(proto['workup']), body_style)]
    ]
    t_referral = Table(referral_rows, colWidths=[110, 150, 110, 150])
    t_referral.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FEF3C7') if int(stage_idx if 'stage_idx' in locals() else 0) >= 2 else colors.HexColor('#F0FDF4')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#F59E0B') if int(stage_idx if 'stage_idx' in locals() else 0) >= 2 else colors.HexColor('#86EFAC')),
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
        [Paragraph("<b>Screening Software:</b> Retina Insight v2.0", body_style),
         Paragraph("<b>Reviewing Ophthalmologist Sign-off:</b> ___________________", body_style)]
    ]
    t_sign = Table(sign_data, colWidths=[280, 240])
    elements.append(t_sign)
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(f"<b>Disclaimer:</b> {html.escape(DISCLAIMER_TEXT)}", disclaimer_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ----------------- SIDEBAR -----------------
st.sidebar.header("📁 Patient Fundus Input")
input_mode = st.sidebar.radio("Input Source:", ["Clinical Database", "Upload Custom Scan"])

img_bgr = None
active_source_label = ""

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

# Patient Clinical Context
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

# Gatekeeper Settings
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Image Quality Settings")
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
        proc_rgb, calibrated_probs, pred_class, overlay = run_inference(img_bgr, temperature)
        confidence = float(calibrated_probs[pred_class] * 100)

        st.markdown("---")
        st.subheader("2. Diagnostic Screening & Reliability")

        res1, res2 = st.columns([1, 1.5])
        with res1:
            st.write(f"### {DIAGNOSIS_INFO[pred_class]['name']}")
            st.progress(min(int(confidence), 100))
            st.write(f"**Calibrated Confidence:** `{confidence:.2f}%`")
            if confidence < LOW_CONFIDENCE_THRESHOLD:
                st.warning("⚠️ Low model confidence. Manual review by a clinician is strongly advised.")
            st.info(f"**Suggested action:** {DIAGNOSIS_INFO[pred_class]['action']}")

        with res2:
            chart_data = {f"Stage {i}": round(float(calibrated_probs[i] * 100), 2) for i in range(5)}
            st.bar_chart(chart_data)

        st.markdown("---")
        st.subheader("3. Explainable AI (XAI) — Grad-CAM Attention Map")
        st.caption("The heatmap shows regions that influenced the model's prediction. It is not a validated lesion detector.")
        v1, v2, v3 = st.columns(3)
        with v1:
            st.image(img_rgb, caption="1. Original Scan", use_container_width=True)
        with v2:
            st.image(proc_rgb, caption="2. Ben Graham Preprocessed", use_container_width=True)
        with v3:
            st.image(overlay, caption="3. Grad-CAM Attention", use_container_width=True)

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
        with st.expander("📍 Find Nearby Eye Hospitals & Retina Specialists (Primary Care Triage)", expanded=(pred_class >= 2)):
            st.markdown("For patients identified with referable or sight-threatening diabetic retinopathy, prompt referral to an accredited vitreo-retinal facility is essential:")
            st.link_button(
                "🔎 Find Nearest Retina Specialists & Eye Hospitals (Google Maps)",
                url="https://www.google.com/maps/search/retina+specialist+eye+hospital+near+me",
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
            "hba1c": p_hba1c
        }

        notes_text, is_gemini, notes_error = generate_clinical_notes(
            stage_idx=pred_class,
            stage_name=DIAGNOSIS_INFO[pred_class]['name'],
            confidence=confidence,
            patient_info=patient_info_dict,
            api_key=gemini_key.strip() if gemini_key else None
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
                stage_idx=pred_class
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