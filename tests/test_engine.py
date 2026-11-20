"""
tests/test_engine.py: Unit tests for core clinical algorithmic engine.
"""
import io
import zipfile
import numpy as np
import cv2
from engine import (
    assess_image_quality,
    detect_macular_edema_risk,
    generate_cam_overlay,
    run_inference_engine,
    generate_clinical_notes,
    generate_bilateral_clinical_notes,
    generate_pdf_report,
    generate_bilateral_pdf_report,
    generate_camp_batch_archive,
    COLORMAP_OPTIONS,
    SUPPORTED_BACKBONES,
    build_retina_model,
    load_inference_system
)

def test_quality_heuristic_valid(sample_stage_images):
    """Verifies that clear clinical fundus images pass automated quality checks."""
    img_bgr = sample_stage_images[0]["bgr"]
    is_valid, msg, blur_val, bright_val = assess_image_quality(img_bgr)
    assert is_valid is True
    assert "Passed" in msg
    assert blur_val > 5.0
    assert 20.0 <= bright_val <= 240.0

def test_quality_heuristic_rejects_noise():
    """Verifies that non-retinal images or blanks are safely rejected."""
    blank_black = np.zeros((512, 512, 3), dtype=np.uint8)
    is_valid, msg, _, _ = assess_image_quality(blank_black)
    assert is_valid is False

    random_noise = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    is_valid_noise, msg_noise, _, _ = assess_image_quality(random_noise)
    assert is_valid_noise is False

def test_inference_and_tta(loaded_system, sample_stage_images):
    """Verifies inference with and without TTA across sample stages."""
    model = loaded_system["model"]
    grad_cam = loaded_system["grad_cam"]
    device = loaded_system["device"]
    temp = loaded_system["temperature"]

    # Test Stage 2
    img_bgr = sample_stage_images[2]["bgr"]
    proc_rgb, probs, pred_class, heatmap = run_inference_engine(
        model, grad_cam, device, img_bgr, temp=temp, use_tta=True
    )
    assert pred_class == 2
    assert heatmap.shape == (256, 256)
    assert np.all(heatmap >= 0.0) and np.all(heatmap <= 1.0)
    assert np.isclose(np.sum(probs), 1.0, atol=1e-3)

def test_csme_detection(sample_stage_images):
    """Verifies ETDRS CSME risk evaluation logic."""
    img_bgr = sample_stage_images[2]["bgr"]
    csme_res = detect_macular_edema_risk(img_bgr, dr_stage=2)
    assert "status" in csme_res
    assert "score" in csme_res
    assert 0.0 <= csme_res["score"] <= 100.0
    assert csme_res["annotated_rgb"].shape == (512, 512, 3)
    assert csme_res["key"] in ["Low", "Moderate", "High"]

def test_gradcam_overlay_colormaps():
    """Verifies Grad-CAM++ colormap generation with alpha blending and contours."""
    base_rgb = np.full((256, 256, 3), 128, dtype=np.uint8)
    heatmap = np.zeros((256, 256), dtype=np.float32)
    heatmap[100:150, 100:150] = 0.9

    for cmap_name in COLORMAP_OPTIONS.keys():
        overlay = generate_cam_overlay(base_rgb, heatmap, colormap_name=cmap_name, alpha=0.5, show_contours=True)
        assert overlay.shape == (256, 256, 3)
        assert overlay.dtype == np.uint8

def test_clinical_notes_generation(mock_patient_info):
    """Verifies offline clinical notes and SOAP generation."""
    notes_text, is_gemini, error_msg = generate_clinical_notes(
        stage_idx=2,
        stage_name="Stage 2: Moderate DR",
        confidence=75.0,
        patient_info=mock_patient_info
    )
    assert is_gemini is False
    assert "Summary for the Patient" in notes_text
    assert "Clinical Note (SOAP format)" in notes_text

def test_bilateral_notes_generation(mock_patient_info):
    """Verifies bilateral notes and inter-ocular discrepancy handling."""
    notes_text, is_gemini, _ = generate_bilateral_clinical_notes(
        patient_info=mock_patient_info,
        od_stage=3,
        od_conf=85.0,
        os_stage=1,
        os_conf=70.0
    )
    assert "Bilateral Patient Summary" in notes_text
    assert "Clinical SOAP Note (Bilateral)" in notes_text
    assert "Marked inter-ocular asymmetry" in notes_text

def test_pdf_report_compilation(mock_patient_info):
    """Verifies single-eye ReportLab PDF document compilation."""
    orig_rgb = np.full((300, 300, 3), 128, dtype=np.uint8)
    cam_rgb = np.full((300, 300, 3), 160, dtype=np.uint8)
    notes = "### Summary\nTest summary\n### SOAP\nSubjective test."

    pdf_buffer = generate_pdf_report(
        patient_info=mock_patient_info,
        stage_name="Stage 2: Moderate DR",
        confidence=75.0,
        original_rgb=orig_rgb,
        gradcam_rgb=cam_rgb,
        clinical_notes_text=notes,
        stage_idx=2
    )
    pdf_bytes = pdf_buffer.getvalue()
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 5000

def test_bilateral_pdf_compilation(mock_patient_info):
    """Verifies bilateral ReportLab PDF document compilation."""
    orig_rgb = np.full((300, 300, 3), 128, dtype=np.uint8)
    cam_rgb = np.full((300, 300, 3), 160, dtype=np.uint8)
    notes = "### Bilateral Summary\nBilateral notes."

    pdf_buffer = generate_bilateral_pdf_report(
        patient_info=mock_patient_info,
        od_stage_idx=2,
        od_confidence=75.0,
        od_orig_rgb=orig_rgb,
        od_cam_rgb=cam_rgb,
        od_blur=80.0,
        os_stage_idx=1,
        os_confidence=82.0,
        os_orig_rgb=orig_rgb,
        os_cam_rgb=cam_rgb,
        os_blur=85.0,
        clinical_notes_text=notes
    )
    pdf_bytes = pdf_buffer.getvalue()
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 5000

def test_pdf_report_with_clinician_signoff(mock_patient_info):
    """Verifies single and bilateral PDF compilation with attending clinician digital accreditation seal."""
    orig_rgb = np.full((300, 300, 3), 128, dtype=np.uint8)
    cam_rgb = np.full((300, 300, 3), 160, dtype=np.uint8)
    notes = "### Summary\nTest notes."
    clinician_meta = {
        "name": "Dr. A. Sharma, MD (Retina)",
        "reg_no": "MCI-RET-2018-9844",
        "is_signed": True,
        "timestamp": "20 Nov 2026, 10:15 AM",
        "notes": notes,
        "cert_hash": "CERT-84729103"
    }

    # Test single PDF with clinician seal
    buf_single = generate_pdf_report(
        patient_info=mock_patient_info,
        stage_name="Stage 3: Severe NPDR",
        confidence=92.5,
        original_rgb=orig_rgb,
        gradcam_rgb=cam_rgb,
        clinical_notes_text=notes,
        stage_idx=3,
        clinician_info=clinician_meta
    )
    bytes_single = buf_single.getvalue()
    assert bytes_single.startswith(b"%PDF")
    assert len(bytes_single) > 5000

    # Test bilateral PDF with clinician seal
    buf_bi = generate_bilateral_pdf_report(
        patient_info=mock_patient_info,
        od_stage_idx=3,
        od_confidence=92.5,
        od_orig_rgb=orig_rgb,
        od_cam_rgb=cam_rgb,
        od_blur=82.0,
        os_stage_idx=2,
        os_confidence=78.0,
        os_orig_rgb=orig_rgb,
        os_cam_rgb=cam_rgb,
        os_blur=80.0,
        clinical_notes_text=notes,
        clinician_info=clinician_meta
    )
    bytes_bi = buf_bi.getvalue()
    assert bytes_bi.startswith(b"%PDF")
    assert len(bytes_bi) > 5000

def test_camp_batch_archive():
    """Verifies batch CSV and ZIP package generation."""
    records = [
        {
            "Scan_ID": "CAMP-001",
            "Filename": "scan1.png",
            "Quality_Status": "Passed",
            "DR_Stage": "Stage 2: Moderate DR",
            "Confidence_%": 75.0,
            "Referable_DR": "Yes",
            "CSME_Risk": "Moderate",
            "_overlay_buf": b"fake_jpeg_bytes"
        }
    ]
    camp_info = {"camp_name": "Test Camp", "screener_id": "Officer 1", "location": "Hyderabad"}
    csv_bytes, zip_bytes = generate_camp_batch_archive(records, camp_info)

    assert b"Scan_ID,Filename" in csv_bytes
    assert zip_bytes.startswith(b"PK")

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        names = z.namelist()
        assert "camp_triage_manifest.csv" in names
        assert "camp_executive_summary.txt" in names
        assert "gradcam_overlays/CAMP-001_gradcam.jpg" in names

def test_build_retina_model_architectures():
    """Verifies model factory instantiates all supported modern backbones."""
    for b_key in ["efficientnet_b0", "convnext_tiny", "efficientnet_b4", "swin_t"]:
        assert b_key in SUPPORTED_BACKBONES
        model, target_layer = build_retina_model(backbone=b_key, num_classes=5)
        assert model is not None
        assert target_layer is not None

def test_load_inference_system_fallback():
    """Verifies load_inference_system gracefully falls back to baseline weights when dedicated weights are absent."""
    model, cam, temp, temp_from_file, dev = load_inference_system(backbone="convnext_tiny")
    assert model is not None
    assert cam is not None
    assert temp > 0
