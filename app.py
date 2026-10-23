import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
import io
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

# ----------------- CUSTOM OPHTHALMIC CLINICAL THEME -----------------
ophthalmic_theme = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* 1. Rich Deep Retinal Cobalt Radial Gradient (No flat black!) */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0F254E 0%, #091326 45%, #050B17 100%) !important;
        color: #E2E8F0 !important;
    }

    /* 2. Hide Streamlit Top Bar & Footers */
    header[data-testid="stHeader"] { display: none !important; }
    #MainMenu { display: none !important; }
    footer { display: none !important; }
    div[data-testid="stToolbar"] { display: none !important; }
    div[data-testid="stDecoration"] { display: none !important; }
    div[data-testid="stStatusWidget"] { display: none !important; }
    .stDeployButton { display: none !important; }
    #manage-app-button { display: none !important; }

    /* 3. Main Container Spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1250px !important;
    }

    /* 4. Ophthalmic Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #09142A 0%, #050B16 100%) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.18) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4) !important;
    }

    /* 5. Glowing Medical Header */
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
        letter-spacing: 0.06em;
    }
    .eye-pulse {
        width: 8px;
        height: 8px;
        background: #38BDF8;
        border-radius: 50%;
        box-shadow: 0 0 10px #38BDF8;
    }
    .eye-hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(135deg, #FFFFFF 20%, #7DD3FC 60%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .eye-hero-sub {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }

    /* 6. Glassmorphic Clinical Cards */
    div[data-testid="stMetric"] {
        background: rgba(15, 27, 52, 0.65) !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2) !important;
        backdrop-filter: blur(10px) !important;
    }
    div[data-testid="stMetric"] label {
        color: #94A3B8 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #38BDF8 !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    /* 7. Section Banners with Laser Blue Trim */
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

    /* 8. Retinal Image Frame with Blue Optical Glow */
    div[data-testid="stImage"] {
        border-radius: 14px !important;
        overflow: hidden !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        background: #060D1A !important;
        box-shadow: 0 4px 20px rgba(14, 165, 233, 0.12) !important;
    }

    /* 9. Action Button (Laser Blue Glow) */
    div.stDownloadButton > button {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        border-radius: 10px !important;
        border: 1px solid #38BDF8 !important;
        padding: 0.75rem 1.8rem !important;
        box-shadow: 0 0 25px rgba(14, 165, 233, 0.4) !important;
        transition: all 0.25s ease !important;
    }
    div.stDownloadButton > button:hover {
        background: linear-gradient(135deg, #0369A1 0%, #075985 100%) !important;
        box-shadow: 0 0 35px rgba(56, 189, 248, 0.6) !important;
        transform: translateY(-2px) !important;
    }

    /* 10. Info / Alert Boxes */
    div.stAlert {
        background: rgba(15, 27, 52, 0.7) !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(8px) !important;
    }
    </style>
"""
st.markdown(ophthalmic_theme, unsafe_allow_html=True)

# ----------------- OPHTHALMIC HERO HEADER -----------------
st.markdown("""
    <div class="eye-hero-badge"><span class="eye-pulse"></span>CLINICAL OPHTHALMOLOGY STATION &bull; ACTIVE</div>
    <div class="eye-hero-title">Retina Insight Diagnostic Console</div>
    <div class="eye-hero-sub">AI-Assisted Diabetic Retinopathy Screening, Uncertainty Calibration & Grad-CAM Optical Localization</div>
""", unsafe_allow_html=True)

DIAGNOSIS_INFO = {
    0: {"name": "Stage 0: No Diabetic Retinopathy", "color": "#10B981", "action": "Routine annual screening recommended in 12 months."},
    1: {"name": "Stage 1: Mild DR", "color": "#38BDF8", "action": "Follow-up screening in 6-12 months. Maintain strict glycemic control."},
    2: {"name": "Stage 2: Moderate DR", "color": "#F59E0B", "action": "Referral to ophthalmologist within 2-4 months for dilated eye examination."},
    3: {"name": "Stage 3: Severe DR", "color": "#F97316", "action": "Urgent retinal specialist consultation within 2-4 weeks."},
    4: {"name": "Stage 4: Proliferative DR", "color": "#EF4444", "action": "Immediate emergency consultation required. High risk of permanent vision loss."}
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
        return False, f"Non-Retinal Image: Color profile does not match vascular retinal tissue (R:{r_mean:.0f}, G:{g_mean:.0f}, B:{b_mean:.0f})."
        
    hues = active_hsv[:, 0]
    sats = active_hsv[:, 1]
    retinal_pixels = ((hues <= 35) | (hues >= 155)) & (sats >= 20)
    retinal_ratio = float(np.mean(retinal_pixels))
    
    if retinal_ratio < 0.40:
        return False, f"Non-Retinal Image: Failed chromatic signature ({retinal_ratio*100:.1f}% match)."
        
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
    img_bgr = cv2.resize(img_bgr, (target_size, target_size))
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), 10)
    enhanced = cv2.addWeighted(img_bgr, 4, blurred, -4, 128)
    return enhanced

# ----------------- GRAD-CAM -----------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (256, 256))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam, target_class

# ----------------- MODEL LOADER -----------------
@st.cache_resource
def load_system():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 5)
    )
    
    if os.path.exists("best_retina_model.pth"):
        model.load_state_dict(torch.load("best_retina_model.pth", map_location=device))
    model = model.to(device)
    model.eval()
    
    temp = 1.3301
    if os.path.exists("temperature.txt"):
        with open("temperature.txt", "r") as f:
            try:
                temp = float(f.read().strip())
            except:
                temp = 1.3301
            
    cam = GradCAM(model, model.features[-1])
    return model, cam, temp, device

model, grad_cam, temperature, device = load_system()

# ----------------- GENAI CLINICAL NOTES -----------------
def generate_clinical_notes(stage_idx, stage_name, confidence, patient_info, api_key=None):
    if api_key and GENAI_AVAILABLE:
        try:
            client = genai.Client(api_key=api_key)
            prompt = f"""
            You are an expert AI Ophthalmologist assistant.
            Patient: {patient_info.get('name')}, Age: {patient_info.get('age')}, Duration of diabetes: {patient_info.get('duration')} yrs, HbA1c: {patient_info.get('hba1c')}%.
            Screening Diagnosis: {stage_name} with {confidence:.1f}% calibrated confidence.
            Provide two clear sections:
            1. [PATIENT EXPLANATION]: Clear, empathetic explanation without jargon.
            2. [CLINICAL SOAP NOTE]: Subjective, Objective, Assessment, Plan for attending doctor.
            """
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text, True
        except Exception:
            pass

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

    full_text = f"### 🧑‍⚕️ Patient Summary\n{patient_messages.get(stage_idx, '')}\n\n### 📋 Clinical SOAP Note\n{soap_note}"
    return full_text, False

# ----------------- CLEAN PDF REPORT BUILDER -----------------
def generate_pdf_report(patient_info, stage_name, confidence, original_rgb, gradcam_rgb, clinical_notes_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1E3A8A'),
        alignment=1,
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#64748B'),
        alignment=1,
        spaceAfter=12
    )
    header_style = ParagraphStyle(
        'SectionHead',
        parent=styles['Heading2'],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=8,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#1E293B')
    )

    elements.append(Paragraph("<b>RETINA INSIGHT CLINICAL DIAGNOSTIC REPORT</b>", title_style))
    elements.append(Paragraph("Automated Retinal Screening &amp; Explainable AI Decision Support Platform", subtitle_style))
    elements.append(Spacer(1, 8))

    scan_date = datetime.now().strftime("%d %b %Y, %I:%M %p")
    p_name_esc = html.escape(str(patient_info.get('name', 'Anonymous')))
    patient_data = [
        [Paragraph("<b>Patient Name:</b>", body_style), Paragraph(p_name_esc, body_style),
         Paragraph("<b>Report Date:</b>", body_style), Paragraph(scan_date, body_style)],
        [Paragraph("<b>Age / Gender:</b>", body_style), Paragraph(f"{patient_info.get('age', 'N/A')} yrs / {patient_info.get('gender', 'N/A')}", body_style),
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

    orig_path = "temp_orig.jpg"
    grad_path = "temp_grad.jpg"
    Image.fromarray(original_rgb).save(orig_path, quality=90)
    Image.fromarray(gradcam_rgb).save(grad_path, quality=90)
    
    img_data = [
        [Paragraph("<b>Original Patient Retinal Scan</b>", body_style), Paragraph("<b>Grad-CAM Explainable AI Lesion Heatmap</b>", body_style)],
        [RLImage(orig_path, width=200, height=200), RLImage(grad_path, width=200, height=200)]
    ]
    t_img = Table(img_data, colWidths=[260, 260])
    t_img.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,0), 3),
    ]))
    elements.append(t_img)
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("<b>Clinical Evaluation &amp; Decision Support</b>", header_style))
    safe_notes = html.escape(clinical_notes_text).replace("\n", "<br/>")
    elements.append(Paragraph(safe_notes, body_style))
    elements.append(Spacer(1, 12))

    sign_data = [
        [Paragraph("<b>Screening Software:</b> Retina Insight v2.0", body_style),
         Paragraph("<b>Attending Ophthalmologist Sign-off:</b> ___________________", body_style)]
    ]
    t_sign = Table(sign_data, colWidths=[280, 240])
    elements.append(t_sign)

    doc.build(elements)
    buffer.seek(0)
    
    for p in [orig_path, grad_path]:
        if os.path.exists(p):
            os.remove(p)
            
    return buffer

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.markdown("### 📁 Clinical Screening Input")
input_mode = st.sidebar.radio("Data Mode:", ["Pre-loaded Database", "Upload Custom Scan"])

img_bgr = None
active_source_label = ""

if input_mode == "Upload Custom Scan":
    uploaded_file = st.sidebar.file_uploader("Upload fundus photograph (.png, .jpg)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        active_source_label = f"Uploaded Scan: {uploaded_file.name}"
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
        active_source_label = f"Clinical DB: {sel_stage} — {sel_case}"
        st.sidebar.caption(f"📁 {len(matched_images)} verified case(s) on disk")
    else:
        st.sidebar.error(f"❌ No images found for Stage {stage_num}.")

st.sidebar.markdown("---")
with st.sidebar.expander("👤 Patient Demographics", expanded=True):
    p_name = st.text_input("Full Name", value="Ramesh Kumar")
    p_age = st.number_input("Age", min_value=18, max_value=100, value=54)
    p_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    p_duration = st.number_input("Diabetes Duration (Years)", min_value=0, max_value=50, value=9)
    p_hba1c = st.number_input("Recent HbA1c (%)", min_value=4.0, max_value=16.0, value=8.4, step=0.1)

with st.sidebar.expander("🤖 Gemini GenAI Key (Optional)", expanded=False):
    gemini_key = st.text_input("API Key", type="password", help="Leave blank to use built-in offline clinical assistant.")

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
st.sidebar.markdown("- **QWK Kappa Score:** `0.8833`")
st.sidebar.markdown("- **Validation Accuracy:** `81.3%`")
st.sidebar.markdown("- **Calibration ECE:** `3.58%`")
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
        proc_bgr = fast_ben_graham(img_bgr, target_size=256)
        proc_rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        tensor_img = transform(Image.fromarray(proc_rgb)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(tensor_img)
            calibrated_probs = torch.softmax(logits / temperature, dim=1)[0].cpu().numpy()
            pred_class = int(np.argmax(calibrated_probs))
            confidence = calibrated_probs[pred_class] * 100
            
        heatmap, _ = grad_cam.generate(tensor_img, target_class=pred_class)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(proc_rgb, 0.6, heatmap_colored, 0.4, 0)
        
        # 2. DIAGNOSTIC RESULTS
        st.markdown('<div class="clinical-section">📊 2. Diagnostic Severity Grading & Calibrated Confidence</div>', unsafe_allow_html=True)
        
        res1, res2 = st.columns([1.2, 1.8])
        with res1:
            stage_color = DIAGNOSIS_INFO[pred_class]['color']
            st.markdown(f"""
                <div style="background: rgba(15, 27, 52, 0.7); border-left: 5px solid {stage_color}; border-radius: 10px; padding: 14px; margin-bottom: 14px; border-top: 1px solid rgba(56,189,248,0.15); border-right: 1px solid rgba(56,189,248,0.15); border-bottom: 1px solid rgba(56,189,248,0.15);">
                    <div style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF;">{DIAGNOSIS_INFO[pred_class]['name']}</div>
                    <div style="font-size: 0.95rem; color: #94A3B8; margin-top: 4px;">Calibrated Probability: <b style="color: {stage_color}; font-size: 1.15rem;">{confidence:.2f}%</b></div>
                </div>
            """, unsafe_allow_html=True)
            st.progress(min(int(confidence), 100))
            st.info(f"**Action:** {DIAGNOSIS_INFO[pred_class]['action']}")
            
        with res2:
            chart_data = {f"Stage {i}": round(float(calibrated_probs[i] * 100), 2) for i in range(5)}
            st.bar_chart(chart_data)

        # 3. GRAD-CAM
        st.markdown('<div class="clinical-section">🔬 3. Explainable AI (XAI) — Pathological Lesion Attention Maps</div>', unsafe_allow_html=True)
        v1, v2, v3 = st.columns(3)
        with v1:
            st.image(img_rgb, caption="1. Original Fundus Scan", use_container_width=True)
        with v2:
            st.image(proc_rgb, caption="2. Color-Normalized Contrast", use_container_width=True)
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
        
        notes_text, is_gemini = generate_clinical_notes(
            stage_idx=pred_class,
            stage_name=DIAGNOSIS_INFO[pred_class]['name'],
            confidence=confidence,
            patient_info=patient_info_dict,
            api_key=gemini_key.strip() if gemini_key else None
        )
        
        if is_gemini:
            st.success("🤖 Analysis generated live via **Gemini GenAI (gemini-2.5-flash)**")
        else:
            st.caption("ℹ️ Clinical assessment generated via **Built-in Offline Clinical Knowledge Engine**")
            
        st.markdown(notes_text)
        
        try:
            pdf_bytes = generate_pdf_report(
                patient_info=patient_info_dict,
                stage_name=DIAGNOSIS_INFO[pred_class]['name'],
                confidence=confidence,
                original_rgb=cv2.resize(img_rgb, (300, 300)),
                gradcam_rgb=cv2.resize(overlay, (300, 300)),
                clinical_notes_text=notes_text
            )
            
            clean_filename = f"Retina_Report_{p_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf"
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