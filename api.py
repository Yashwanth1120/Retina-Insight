"""
api.py: Retina Insight Production-Ready FastAPI Microservice
Provides headless RESTful endpoints for automated diabetic retinopathy grading,
Grad-CAM++ visual explanations, ETDRS CSME risk stratification, bilateral (OD vs OS) triage,
and hospital-grade PDF report exports.
"""
import time
import base64
import os
import re
import cv2
import numpy as np
from contextlib import asynccontextmanager
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from engine import (
    NUM_CLASSES,
    DEFAULT_TEMPERATURE,
    LOW_CONFIDENCE_THRESHOLD,
    BENCHMARKS,
    EXTERNAL,
    CLINICAL_PROTOCOLS,
    DIAGNOSIS_INFO,
    CSME_INFO,
    COLORMAP_OPTIONS,
    assess_image_quality,
    load_inference_system,
    run_inference_engine,
    generate_cam_overlay,
    detect_macular_edema_risk,
    generate_camp_batch_archive,
    generate_clinical_notes,
    generate_bilateral_clinical_notes,
    generate_pdf_report,
    generate_bilateral_pdf_report
)

START_TIME = time.time()
SYSTEM_STATE = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load PyTorch model weights, Grad-CAM++, and temperature
    model, grad_cam, temp, temp_from_file, device = load_inference_system(
        model_path="best_retina_model.pth",
        temp_path="temperature.txt"
    )
    SYSTEM_STATE["model"] = model
    SYSTEM_STATE["grad_cam"] = grad_cam
    SYSTEM_STATE["temperature"] = temp
    SYSTEM_STATE["device"] = device
    SYSTEM_STATE["temp_from_file"] = temp_from_file
    yield
    # Shutdown
    SYSTEM_STATE.clear()

app = FastAPI(
    title="Retina Insight — Clinical Decision Support API",
    description=(
        "Production-grade ophthalmic AI microservice providing automated 5-stage Diabetic Retinopathy "
        "grading, Grad-CAM++ lesion localization, ETDRS Clinically Significant Macular Edema (CSME) "
        "risk detection, bilateral OD/OS triage, and EHR/PDF report generation."
    ),
    version="2.5.0",
    lifespan=lifespan
)

# CORS Middleware for integration with HIS, PACS, and web applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ----------------- PYDANTIC SCHEMAS -----------------
class HealthCheckResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    model: str = Field(..., examples=["EfficientNet-B0 + 5-Class Head"])
    device: str = Field(..., examples=["cpu"])
    calibration_temperature: float = Field(..., examples=[1.3782])
    uptime_seconds: float = Field(..., examples=[120.4])
    version: str = Field(..., examples=["2.5.0"])

class SystemBenchmarksResponse(BaseModel):
    aptos_validation_split: dict
    idrid_external_test: dict
    calibration_temperature: float
    confidence_alert_threshold: float

class QualityReport(BaseModel):
    is_valid: bool = Field(..., examples=[True])
    message: str = Field(..., examples=["Passed: image quality acceptable"])
    blur_score: float = Field(..., examples=[79.7])
    brightness: float = Field(..., examples=[125.4])

class StageProbability(BaseModel):
    stage: int = Field(..., examples=[2])
    name: str = Field(..., examples=["Stage 2: Moderate DR"])
    probability_pct: float = Field(..., examples=[73.0])

class CSMEReport(BaseModel):
    risk_level: str = Field(..., examples=["Moderate"])
    status: str = Field(..., examples=["CSME Equivocal / Moderate Risk"])
    score: float = Field(..., examples=[35.3])
    badge_color: str = Field(..., examples=["orange"])
    foveal_exudates_px: int = Field(..., examples=[0])
    macular_exudates_px: int = Field(..., examples=[1413])
    foveal_attention_intensity: float = Field(..., examples=[0.15])
    etdrs_finding: str = Field(..., examples=["Hard exudates in macular periphery"])
    clinical_directive: str = Field(..., examples=["OCT Macula recommended within 4-6 weeks"])

class ClinicalCarePlan(BaseModel):
    triage_urgency: str = Field(..., examples=["Referable Diabetic Retinopathy (Moderate Risk)"])
    timeline: str = Field(..., examples=["Specialist referral within 2 to 3 months"])
    referral_level: str = Field(..., examples=["Comprehensive Ophthalmologist / Retina Clinic"])
    diagnostic_workup: str = Field(..., examples=["OCT Macula + Dilated Fundus Examination"])
    target_goals: List[str]

class ScreenResult(BaseModel):
    scan_id: str = Field(..., examples=["SCAN-001"])
    quality: QualityReport
    predicted_stage: int = Field(..., examples=[2])
    stage_name: str = Field(..., examples=["Stage 2: Moderate DR"])
    calibrated_confidence_pct: float = Field(..., examples=[73.0])
    low_confidence_flag: bool = Field(..., examples=[False])
    referable_dr: bool = Field(..., examples=[True])
    stage_probabilities: List[StageProbability]
    csme: CSMEReport
    care_plan: ClinicalCarePlan
    cam_overlay_base64: Optional[str] = None
    csme_annotated_base64: Optional[str] = None

class BilateralScreenResult(BaseModel):
    od: ScreenResult
    os: ScreenResult
    overall_stage: int = Field(..., examples=[2])
    overall_stage_name: str = Field(..., examples=["Stage 2: Moderate DR"])
    governing_eye: str = Field(..., examples=["Right Eye (OD)"])
    asymmetry_stage_delta: int = Field(..., examples=[1])
    asymmetry_classification: str = Field(..., examples=["Mild Inter-Ocular Asymmetry"])
    clinical_red_flag: bool = Field(..., examples=[False])
    overall_urgency: str = Field(..., examples=["Referable Diabetic Retinopathy (Moderate Risk)"])
    overall_timeline: str = Field(..., examples=["Specialist referral within 2 to 3 months"])
    recommended_action: str

class BatchManifestRecord(BaseModel):
    scan_id: str
    filename: str
    quality_status: str
    dr_stage: str
    confidence_pct: float
    referable: str
    csme_risk: str
    csme_score: float
    triage_urgency: str
    follow_up: str
    action: str

class BatchScreenResult(BaseModel):
    total_scanned: int
    quality_passed: int
    quality_rejected: int
    referable_dr_count: int
    sight_threatening_count: int
    csme_alert_count: int
    manifest: List[BatchManifestRecord]
    executive_summary: str

# ----------------- HELPER FUNCTIONS -----------------
def _decode_image_bytes(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None or img_bgr.size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid or readable image."
        )
    return img_bgr

def _encode_b64(rgb_array: np.ndarray, ext: str = ".jpg") -> str:
    bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    is_ok, buffer = cv2.imencode(ext, bgr)
    if not is_ok:
        return ""
    return base64.b64encode(buffer).decode("utf-8")

# ----------------- CORE API ENDPOINTS -----------------
@app.get("/", summary="API Root & Status", tags=["General"])
async def root():
    """Returns microservice identity and link to interactive OpenAPI Swagger docs."""
    return {
        "service": "Retina Insight — Clinical Decision Support API",
        "version": "2.5.0",
        "documentation": "/docs",
        "endpoints": [
            "/health",
            "/api/v1/benchmarks",
            "/api/v1/screen",
            "/api/v1/screen-bilateral",
            "/api/v1/screen-batch",
            "/api/v1/report/pdf",
            "/api/v1/report/bilateral-pdf",
            "/api/v1/report/camp-archive"
        ]
    }

@app.get("/health", response_model=HealthCheckResponse, summary="Readiness & Liveness Probe", tags=["General"])
async def health_check():
    """Checks whether model weights, calibration factors, and device are loaded and ready."""
    model = SYSTEM_STATE.get("model")
    device = SYSTEM_STATE.get("device", "cpu")
    temp = SYSTEM_STATE.get("temperature", DEFAULT_TEMPERATURE)

    if model is None:
        raise HTTPException(status_code=503, detail="Model weights are not loaded.")

    return HealthCheckResponse(
        status="healthy",
        model="EfficientNet-B0 + 5-Class Head",
        device=str(device),
        calibration_temperature=float(temp),
        uptime_seconds=round(time.time() - START_TIME, 2),
        version="2.5.0"
    )

@app.get("/api/v1/benchmarks", response_model=SystemBenchmarksResponse, summary="Model Benchmarks & Calibration", tags=["Clinical"])
async def get_benchmarks():
    """Returns validated internal APTOS and external IDRiD diagnostic benchmarks."""
    return SystemBenchmarksResponse(
        aptos_validation_split=BENCHMARKS,
        idrid_external_test=EXTERNAL,
        calibration_temperature=SYSTEM_STATE.get("temperature", DEFAULT_TEMPERATURE),
        confidence_alert_threshold=LOW_CONFIDENCE_THRESHOLD
    )

@app.post("/api/v1/screen", response_model=ScreenResult, summary="Single Retinal Fundus Screening", tags=["Clinical"])
async def screen_single_fundus(
    file: UploadFile = File(..., description="Fundus photograph image (.png, .jpg, .jpeg)"),
    scan_id: str = Query("SCAN-001", description="Unique clinic / study accession ID"),
    use_tta: bool = Query(True, description="Enable 4-fold rotational/flip Test-Time Augmentation"),
    multi_scale: bool = Query(False, description="Enable multi-scale center zoom crop ensemble"),
    include_overlay: bool = Query(False, description="Return base64-encoded Grad-CAM++ JPEG overlay"),
    include_csme: bool = Query(False, description="Return base64-encoded CSME annotated macular image"),
    blur_threshold: float = Query(5.0, description="Minimum standardized blur threshold")
):
    """
    Automated diabetic retinopathy grading on a single fundus photograph:
    - Pre-screening blur & illumination heuristics
    - EfficientNet-B0 inference with temperature calibration & TTA
    - ETDRS-standard Clinically Significant Macular Edema (CSME) risk detection
    - Grad-CAM++ lesion localization
    """
    contents = await file.read()
    img_bgr = _decode_image_bytes(contents)

    is_valid, q_msg, blur_score, brightness = assess_image_quality(img_bgr, blur_threshold=blur_threshold)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image failed automated quality check: {q_msg} (Blur: {blur_score:.1f}, Brightness: {brightness:.1f})"
        )

    model = SYSTEM_STATE["model"]
    grad_cam = SYSTEM_STATE["grad_cam"]
    device = SYSTEM_STATE["device"]
    temp = SYSTEM_STATE["temperature"]

    proc_rgb, probs, pred_class, heatmap = run_inference_engine(
        model, grad_cam, device, img_bgr, temp=temp, use_tta=use_tta, multi_scale=multi_scale
    )
    conf = float(probs[pred_class] * 100)
    csme_res = detect_macular_edema_risk(img_bgr, heatmap=heatmap, dr_stage=pred_class)
    proto = CLINICAL_PROTOCOLS[pred_class]
    is_referable = bool(pred_class >= 2 or csme_res["key"] in ["Moderate", "High"])

    cam_b64 = None
    if include_overlay:
        overlay = generate_cam_overlay(proc_rgb, heatmap, colormap_name="JET (Classic Ophthalmology)", alpha=0.45)
        cam_b64 = _encode_b64(overlay)

    csme_b64 = None
    if include_csme:
        csme_b64 = _encode_b64(csme_res["annotated_rgb"])

    stage_probs = [
        StageProbability(
            stage=i,
            name=DIAGNOSIS_INFO[i]["name"],
            probability_pct=round(float(probs[i] * 100), 2)
        )
        for i in range(NUM_CLASSES)
    ]

    return ScreenResult(
        scan_id=scan_id,
        quality=QualityReport(
            is_valid=is_valid,
            message=q_msg,
            blur_score=round(blur_score, 1),
            brightness=round(brightness, 1)
        ),
        predicted_stage=pred_class,
        stage_name=DIAGNOSIS_INFO[pred_class]["name"],
        calibrated_confidence_pct=round(conf, 2),
        low_confidence_flag=bool(conf < LOW_CONFIDENCE_THRESHOLD),
        referable_dr=is_referable,
        stage_probabilities=stage_probs,
        csme=CSMEReport(
            risk_level=csme_res["key"],
            status=csme_res["status"],
            score=round(float(csme_res["score"]), 1),
            badge_color=csme_res["badge_color"],
            foveal_exudates_px=csme_res["foveal_exudates"],
            macular_exudates_px=csme_res["macular_exudates"],
            foveal_attention_intensity=round(float(csme_res["foveal_cam_intensity"]), 3),
            etdrs_finding=csme_res["desc"],
            clinical_directive=csme_res["action"]
        ),
        care_plan=ClinicalCarePlan(
            triage_urgency=proto["urgency"],
            timeline=proto["timeline"],
            referral_level=proto["referral_level"],
            diagnostic_workup=proto["workup"],
            target_goals=proto["target_goals"]
        ),
        cam_overlay_base64=cam_b64,
        csme_annotated_base64=csme_b64
    )

@app.post("/api/v1/screen-bilateral", response_model=BilateralScreenResult, summary="Bilateral (OD vs OS) Screening", tags=["Clinical"])
async def screen_bilateral(
    file_od: UploadFile = File(..., description="Right Eye (OD — Oculus Dexter) fundus photograph"),
    file_os: UploadFile = File(..., description="Left Eye (OS — Oculus Sinister) fundus photograph"),
    use_tta: bool = Query(True, description="Enable 4-fold Test-Time Augmentation"),
    multi_scale: bool = Query(False, description="Enable multi-scale center crop ensemble"),
    include_overlay: bool = Query(False, description="Return base64-encoded Grad-CAM++ JPEG overlays")
):
    """
    Simultaneous bilateral multi-eye examination:
    - Evaluates Right Eye (OD) and Left Eye (OS)
    - Detects inter-ocular stage asymmetry (Δ = |OD - OS|)
    - Flags marked discrepancy (Δ >= 2) for carotid Doppler ultrasonography
    - Overall patient triage urgency is governed strictly by the higher-severity eye
    """
    contents_od = await file_od.read()
    contents_os = await file_os.read()

    img_bgr_od = _decode_image_bytes(contents_od)
    img_bgr_os = _decode_image_bytes(contents_os)

    is_valid_od, msg_od, blur_od, bright_od = assess_image_quality(img_bgr_od)
    is_valid_os, msg_os, blur_os, bright_os = assess_image_quality(img_bgr_os)

    if not is_valid_od or not is_valid_os:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Bilateral quality failure. OD: {msg_od} | OS: {msg_os}"
        )

    model = SYSTEM_STATE["model"]
    grad_cam = SYSTEM_STATE["grad_cam"]
    device = SYSTEM_STATE["device"]
    temp = SYSTEM_STATE["temperature"]

    proc_od, probs_od, pred_od, cam_od = run_inference_engine(model, grad_cam, device, img_bgr_od, temp=temp, use_tta=use_tta, multi_scale=multi_scale)
    proc_os, probs_os, pred_os, cam_os = run_inference_engine(model, grad_cam, device, img_bgr_os, temp=temp, use_tta=use_tta, multi_scale=multi_scale)

    conf_od = float(probs_od[pred_od] * 100)
    conf_os = float(probs_os[pred_os] * 100)

    csme_od = detect_macular_edema_risk(img_bgr_od, heatmap=cam_od, dr_stage=pred_od)
    csme_os = detect_macular_edema_risk(img_bgr_os, heatmap=cam_os, dr_stage=pred_os)

    overall_stage = max(pred_od, pred_os)
    diff = abs(pred_od - pred_os)
    proto_overall = CLINICAL_PROTOCOLS[overall_stage]
    higher_eye = "Right Eye (OD)" if pred_od > pred_os else "Left Eye (OS)" if pred_os > pred_od else "Bilateral Concordant (OD & OS)"

    if diff == 0:
        asym_class = "Bilateral Concordant (Symmetric Involvement)"
        red_flag = False
        action = "Routine bilateral monitoring per standard protocol."
    elif diff == 1:
        asym_class = "Mild Inter-Ocular Asymmetry (Δ = 1 Stage)"
        red_flag = False
        action = f"Triage urgency governed by {higher_eye}. Monitor contralateral eye closely."
    else:
        asym_class = f"CLINICAL RED FLAG: Marked Inter-Ocular Asymmetry (Δ = {diff} Stages)"
        red_flag = True
        action = "Urgent vitreo-retinal consultation. Recommend carotid duplex Doppler ultrasonography to rule out ocular ischemic syndrome."

    cam_od_b64 = _encode_b64(generate_cam_overlay(proc_od, cam_od)) if include_overlay else None
    cam_os_b64 = _encode_b64(generate_cam_overlay(proc_os, cam_os)) if include_overlay else None

    res_od = ScreenResult(
        scan_id="OD-RightEye",
        quality=QualityReport(is_valid=is_valid_od, message=msg_od, blur_score=round(blur_od, 1), brightness=round(bright_od, 1)),
        predicted_stage=pred_od,
        stage_name=DIAGNOSIS_INFO[pred_od]["name"],
        calibrated_confidence_pct=round(conf_od, 2),
        low_confidence_flag=bool(conf_od < LOW_CONFIDENCE_THRESHOLD),
        referable_dr=bool(pred_od >= 2 or csme_od["key"] in ["Moderate", "High"]),
        stage_probabilities=[StageProbability(stage=i, name=DIAGNOSIS_INFO[i]["name"], probability_pct=round(float(probs_od[i]*100), 2)) for i in range(5)],
        csme=CSMEReport(
            risk_level=csme_od["key"], status=csme_od["status"], score=round(float(csme_od["score"]), 1),
            badge_color=csme_od["badge_color"], foveal_exudates_px=csme_od["foveal_exudates"],
            macular_exudates_px=csme_od["macular_exudates"], foveal_attention_intensity=round(float(csme_od["foveal_cam_intensity"]), 3),
            etdrs_finding=csme_od["desc"], clinical_directive=csme_od["action"]
        ),
        care_plan=ClinicalCarePlan(
            triage_urgency=CLINICAL_PROTOCOLS[pred_od]["urgency"], timeline=CLINICAL_PROTOCOLS[pred_od]["timeline"],
            referral_level=CLINICAL_PROTOCOLS[pred_od]["referral_level"], diagnostic_workup=CLINICAL_PROTOCOLS[pred_od]["workup"],
            target_goals=CLINICAL_PROTOCOLS[pred_od]["target_goals"]
        ),
        cam_overlay_base64=cam_od_b64
    )

    res_os = ScreenResult(
        scan_id="OS-LeftEye",
        quality=QualityReport(is_valid=is_valid_os, message=msg_os, blur_score=round(blur_os, 1), brightness=round(bright_os, 1)),
        predicted_stage=pred_os,
        stage_name=DIAGNOSIS_INFO[pred_os]["name"],
        calibrated_confidence_pct=round(conf_os, 2),
        low_confidence_flag=bool(conf_os < LOW_CONFIDENCE_THRESHOLD),
        referable_dr=bool(pred_os >= 2 or csme_os["key"] in ["Moderate", "High"]),
        stage_probabilities=[StageProbability(stage=i, name=DIAGNOSIS_INFO[i]["name"], probability_pct=round(float(probs_os[i]*100), 2)) for i in range(5)],
        csme=CSMEReport(
            risk_level=csme_os["key"], status=csme_os["status"], score=round(float(csme_os["score"]), 1),
            badge_color=csme_os["badge_color"], foveal_exudates_px=csme_os["foveal_exudates"],
            macular_exudates_px=csme_os["macular_exudates"], foveal_attention_intensity=round(float(csme_os["foveal_cam_intensity"]), 3),
            etdrs_finding=csme_os["desc"], clinical_directive=csme_os["action"]
        ),
        care_plan=ClinicalCarePlan(
            triage_urgency=CLINICAL_PROTOCOLS[pred_os]["urgency"], timeline=CLINICAL_PROTOCOLS[pred_os]["timeline"],
            referral_level=CLINICAL_PROTOCOLS[pred_os]["referral_level"], diagnostic_workup=CLINICAL_PROTOCOLS[pred_os]["workup"],
            target_goals=CLINICAL_PROTOCOLS[pred_os]["target_goals"]
        ),
        cam_overlay_base64=cam_os_b64
    )

    return BilateralScreenResult(
        od=res_od,
        os=res_os,
        overall_stage=overall_stage,
        overall_stage_name=DIAGNOSIS_INFO[overall_stage]["name"],
        governing_eye=higher_eye,
        asymmetry_stage_delta=diff,
        asymmetry_classification=asym_class,
        clinical_red_flag=red_flag,
        overall_urgency=proto_overall["urgency"],
        overall_timeline=proto_overall["timeline"],
        recommended_action=action
    )

@app.post("/api/v1/screen-batch", response_model=BatchScreenResult, summary="Screening Camp Batch Triage", tags=["Clinical"])
async def screen_batch(
    files: List[UploadFile] = File(..., description="Batch of retinal fundus photographs"),
    camp_name: str = Query("Community Vision Screening Camp", description="Screening facility or camp name"),
    screener_id: str = Query("Dr. Field Officer", description="Screening officer or clinician identifier"),
    location: str = Query("Rural Outreach Camp", description="Camp district or location")
):
    """
    High-throughput batch triage for community screening camps & mobile health vans:
    - Filters unassessable blurred images
    - Grades all valid fundus scans with TTA
    - Compiles epidemiological statistics and filterable EHR triage manifest
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded in batch.")

    model = SYSTEM_STATE["model"]
    grad_cam = SYSTEM_STATE["grad_cam"]
    device = SYSTEM_STATE["device"]
    temp = SYSTEM_STATE["temperature"]

    manifest = []
    total_scanned = len(files)
    quality_passed = 0
    referable_count = 0
    sight_threatening_count = 0
    csme_alert_count = 0

    for idx, f in enumerate(files):
        scan_id = f"CAMP-{idx+1:03d}"
        contents = await f.read()
        try:
            im_bgr = _decode_image_bytes(contents)
            is_valid, q_msg, blur_v, bright_v = assess_image_quality(im_bgr)
        except Exception:
            is_valid, q_msg, blur_v, bright_v = False, "Corrupted Image Data", 0.0, 0.0

        if not is_valid:
            manifest.append(BatchManifestRecord(
                scan_id=scan_id,
                filename=f.filename or scan_id,
                quality_status="Rejected",
                dr_stage="Unusable (Quality Failed)",
                confidence_pct=0.0,
                referable="Re-scan Required",
                csme_risk="Not Evaluated",
                csme_score=0.0,
                triage_urgency="Immediate Re-scan Required",
                follow_up="Same Day (Re-take)",
                action="Immediate camera re-acquisition required on site"
            ))
        else:
            quality_passed += 1
            proc_rgb, probs, pred_c, heatmap = run_inference_engine(model, grad_cam, device, im_bgr, temp=temp, use_tta=True)
            conf = float(probs[pred_c] * 100)
            csme_res = detect_macular_edema_risk(im_bgr, heatmap=heatmap, dr_stage=pred_c)
            proto = CLINICAL_PROTOCOLS[pred_c]
            is_ref = bool(pred_c >= 2 or csme_res["key"] in ["Moderate", "High"])

            if is_ref:
                referable_count += 1
            if pred_c in [3, 4]:
                sight_threatening_count += 1
            if csme_res["key"] in ["Moderate", "High"]:
                csme_alert_count += 1

            manifest.append(BatchManifestRecord(
                scan_id=scan_id,
                filename=f.filename or scan_id,
                quality_status="Passed",
                dr_stage=DIAGNOSIS_INFO[pred_c]["name"],
                confidence_pct=round(conf, 1),
                referable="Yes (Referable)" if is_ref else "No (Routine)",
                csme_risk=csme_res["name"],
                csme_score=round(float(csme_res["score"]), 1),
                triage_urgency=proto["urgency"],
                follow_up=proto["timeline"],
                action=f"{proto['urgency']} — {proto['referral_level']}"
            ))

    summary_text = (
        f"RETINA INSIGHT CAMP SUMMARY: {camp_name} ({location})\n"
        f"Total Ingested: {total_scanned} | Passed: {quality_passed} | Rejects: {total_scanned - quality_passed}\n"
        f"Referable DR Cases: {referable_count} | Sight-Threatening: {sight_threatening_count} | CSME Alerts: {csme_alert_count}"
    )

    return BatchScreenResult(
        total_scanned=total_scanned,
        quality_passed=quality_passed,
        quality_rejected=total_scanned - quality_passed,
        referable_dr_count=referable_count,
        sight_threatening_count=sight_threatening_count,
        csme_alert_count=csme_alert_count,
        manifest=manifest,
        executive_summary=summary_text
    )

# ----------------- PDF & ARCHIVE EXPORT ENDPOINTS -----------------
@app.post("/api/v1/report/pdf", summary="Compile Single-Eye PDF Report", tags=["Reports"])
async def export_single_pdf(
    file: UploadFile = File(..., description="Fundus photograph"),
    patient_name: str = Form("Ramesh Kumar"),
    age: int = Form(54),
    gender: str = Form("Male"),
    duration_years: int = Form(9),
    hba1c: float = Form(8.4),
    location: str = Form("Hyderabad"),
    clinician_name: Optional[str] = Form(None),
    clinician_reg_no: Optional[str] = Form(None)
):
    """Compiles and streams a clinical ReportLab PDF report for single fundus screening."""
    contents = await file.read()
    img_bgr = _decode_image_bytes(contents)

    model = SYSTEM_STATE["model"]
    grad_cam = SYSTEM_STATE["grad_cam"]
    device = SYSTEM_STATE["device"]
    temp = SYSTEM_STATE["temperature"]

    proc_rgb, probs, pred_class, heatmap = run_inference_engine(model, grad_cam, device, img_bgr, temp=temp, use_tta=True)
    conf = float(probs[pred_class] * 100)
    csme_res = detect_macular_edema_risk(img_bgr, heatmap=heatmap, dr_stage=pred_class)
    overlay = generate_cam_overlay(proc_rgb, heatmap, colormap_name="JET (Classic Ophthalmology)", alpha=0.45)

    pt_info = {
        "name": patient_name,
        "age": age,
        "gender": gender,
        "duration": duration_years,
        "hba1c": hba1c,
        "location": location
    }

    notes_text, _, _ = generate_clinical_notes(
        stage_idx=pred_class,
        stage_name=DIAGNOSIS_INFO[pred_class]["name"],
        confidence=conf,
        patient_info=pt_info,
        csme_info=csme_res
    )

    c_info = None
    if clinician_name and clinician_name.strip():
        c_info = {
            "name": clinician_name.strip(),
            "reg_no": (clinician_reg_no.strip() if clinician_reg_no else "MCI-RET-VALIDATED"),
            "is_signed": True,
            "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p")
        }

    pdf_buffer = generate_pdf_report(
        patient_info=pt_info,
        stage_name=DIAGNOSIS_INFO[pred_class]["name"],
        confidence=conf,
        original_rgb=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        gradcam_rgb=overlay,
        clinical_notes_text=notes_text,
        stage_idx=pred_class,
        csme_info=csme_res,
        clinician_info=c_info
    )

    pdf_bytes = pdf_buffer.getvalue()
    safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', patient_name).strip() or "Patient"
    filename = f"Retina_Report_{safe_name}_{datetime.now().strftime('%Y%m%d')}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.post("/api/v1/report/bilateral-pdf", summary="Compile Bilateral PDF Report", tags=["Reports"])
async def export_bilateral_pdf(
    file_od: UploadFile = File(..., description="Right Eye (OD) fundus photograph"),
    file_os: UploadFile = File(..., description="Left Eye (OS) fundus photograph"),
    patient_name: str = Form("Ramesh Kumar"),
    age: int = Form(54),
    gender: str = Form("Male"),
    duration_years: int = Form(9),
    hba1c: float = Form(8.4),
    location: str = Form("Hyderabad"),
    clinician_name: Optional[str] = Form(None),
    clinician_reg_no: Optional[str] = Form(None)
):
    """Compiles and streams a bilateral (OD vs OS) ReportLab PDF report."""
    contents_od = await file_od.read()
    contents_os = await file_os.read()

    img_od = _decode_image_bytes(contents_od)
    img_os = _decode_image_bytes(contents_os)

    is_valid_od, _, blur_od, _ = assess_image_quality(img_od)
    is_valid_os, _, blur_os, _ = assess_image_quality(img_os)

    model = SYSTEM_STATE["model"]
    grad_cam = SYSTEM_STATE["grad_cam"]
    device = SYSTEM_STATE["device"]
    temp = SYSTEM_STATE["temperature"]

    proc_od, probs_od, pred_od, cam_od = run_inference_engine(model, grad_cam, device, img_od, temp=temp, use_tta=True)
    proc_os, probs_os, pred_os, cam_os = run_inference_engine(model, grad_cam, device, img_os, temp=temp, use_tta=True)

    conf_od = float(probs_od[pred_od] * 100)
    conf_os = float(probs_os[pred_os] * 100)

    csme_od = detect_macular_edema_risk(img_od, heatmap=cam_od, dr_stage=pred_od)
    csme_os = detect_macular_edema_risk(img_os, heatmap=cam_os, dr_stage=pred_os)

    overlay_od = generate_cam_overlay(proc_od, cam_od)
    overlay_os = generate_cam_overlay(proc_os, cam_os)

    pt_info = {
        "name": patient_name,
        "age": age,
        "gender": gender,
        "duration": duration_years,
        "hba1c": hba1c,
        "location": location
    }

    notes_text, _, _ = generate_bilateral_clinical_notes(
        patient_info=pt_info,
        od_stage=pred_od,
        od_conf=conf_od,
        os_stage=pred_os,
        os_conf=conf_os,
        od_csme=csme_od,
        os_csme=csme_os
    )

    c_info = None
    if clinician_name and clinician_name.strip():
        c_info = {
            "name": clinician_name.strip(),
            "reg_no": (clinician_reg_no.strip() if clinician_reg_no else "MCI-RET-VALIDATED"),
            "is_signed": True,
            "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p")
        }

    pdf_buffer = generate_bilateral_pdf_report(
        patient_info=pt_info,
        od_stage_idx=pred_od,
        od_confidence=conf_od,
        od_orig_rgb=cv2.cvtColor(img_od, cv2.COLOR_BGR2RGB),
        od_cam_rgb=overlay_od,
        od_blur=blur_od,
        os_stage_idx=pred_os,
        os_confidence=conf_os,
        os_orig_rgb=cv2.cvtColor(img_os, cv2.COLOR_BGR2RGB),
        os_cam_rgb=overlay_os,
        os_blur=blur_os,
        clinical_notes_text=notes_text,
        od_csme=csme_od,
        os_csme=csme_os,
        clinician_info=c_info
    )

    pdf_bytes = pdf_buffer.getvalue()
    safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', patient_name).strip() or "Patient"
    filename = f"Retina_Bilateral_Report_{safe_name}_{datetime.now().strftime('%Y%m%d')}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.post("/api/v1/report/camp-archive", summary="Generate Complete Camp ZIP Archive", tags=["Reports"])
async def export_camp_archive(
    files: List[UploadFile] = File(..., description="Batch of retinal fundus photographs"),
    camp_name: str = Query("Community Vision Screening Camp", description="Screening camp name"),
    screener_id: str = Query("Dr. Field Officer", description="Screening clinician identifier"),
    location: str = Query("Rural Outreach Camp", description="Camp district or location")
):
    """
    Processes screening camp batch and streams a complete in-memory ZIP archive
    containing CSV manifest, executive summary, and Grad-CAM++ overlays.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded in batch.")

    model = SYSTEM_STATE["model"]
    grad_cam = SYSTEM_STATE["grad_cam"]
    device = SYSTEM_STATE["device"]
    temp = SYSTEM_STATE["temperature"]

    batch_records = []
    camp_info = {"camp_name": camp_name, "screener_id": screener_id, "location": location}

    for idx, f in enumerate(files):
        scan_id = f"CAMP-{idx+1:03d}"
        contents = await f.read()
        try:
            im_bgr = _decode_image_bytes(contents)
            is_valid, q_msg, blur_v, bright_v = assess_image_quality(im_bgr)
        except Exception:
            is_valid, q_msg, blur_v, bright_v = False, "Corrupted image", 0.0, 0.0

        if not is_valid:
            batch_records.append({
                "Scan_ID": scan_id,
                "Filename": f.filename or scan_id,
                "Quality_Status": "Rejected",
                "Quality_Notes": q_msg,
                "Blur_Score": round(float(blur_v), 1),
                "Brightness": round(float(bright_v), 1),
                "DR_Stage": "Unusable",
                "Confidence_%": 0.0,
                "Referable_DR": "Re-scan Required",
                "CSME_Risk": "Not Evaluated",
                "CSME_Score": 0.0,
                "Triage_Urgency": "Immediate Re-scan Required",
                "Follow_up_Timeline": "Immediate",
                "Recommended_Action": "Immediate camera re-acquisition required on site"
            })
        else:
            proc_rgb, probs, pred_c, heatmap = run_inference_engine(model, grad_cam, device, im_bgr, temp=temp, use_tta=True)
            conf = float(probs[pred_c] * 100)
            csme_res = detect_macular_edema_risk(im_bgr, heatmap=heatmap, dr_stage=pred_c)
            proto = CLINICAL_PROTOCOLS[pred_c]
            is_ref = "Yes (Referable)" if (pred_c >= 2 or csme_res["key"] in ["Moderate", "High"]) else "No (Routine)"

            overlay = generate_cam_overlay(proc_rgb, heatmap)
            is_ok, buf = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            overlay_bytes = buf.tobytes() if is_ok else None

            batch_records.append({
                "Scan_ID": scan_id,
                "Filename": f.filename or scan_id,
                "Quality_Status": "Passed",
                "Quality_Notes": q_msg,
                "Blur_Score": round(float(blur_v), 1),
                "Brightness": round(float(bright_v), 1),
                "DR_Stage": DIAGNOSIS_INFO[pred_c]["name"],
                "Confidence_%": round(conf, 1),
                "Referable_DR": is_ref,
                "CSME_Risk": csme_res["name"],
                "CSME_Score": round(float(csme_res["score"]), 1),
                "Triage_Urgency": proto["urgency"],
                "Follow_up_Timeline": proto["timeline"],
                "Recommended_Action": f"{proto['urgency']} — {proto['referral_level']}",
                "_overlay_buf": overlay_bytes
            })

    csv_bytes, zip_bytes = generate_camp_batch_archive(batch_records, camp_info)
    camp_slug = re.sub(r'[^A-Za-z0-9_-]+', '_', camp_name).strip() or "Camp"
    zip_filename = f"Retina_Camp_Archive_{camp_slug}_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
    )
