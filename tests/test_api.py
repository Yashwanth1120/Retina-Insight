"""
tests/test_api.py: Integration tests for FastAPI endpoints.
"""
import io
import base64
import zipfile
import pytest

def test_api_root(client):
    """Verifies that root returns service info and endpoints."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Retina Insight" in data["service"]
    assert "/health" in data["endpoints"]

def test_api_health(client):
    """Verifies that /health returns healthy status, model details, and calibration temp."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "EfficientNet" in data["model"]
    assert data["calibration_temperature"] > 0

def test_api_benchmarks(client):
    """Verifies /api/v1/benchmarks returns validated metrics."""
    response = client.get("/api/v1/benchmarks")
    assert response.status_code == 200
    data = response.json()
    assert "qwk" in data["aptos_validation_split"]
    assert "auc_referable" in data["idrid_external_test"]

def test_screen_single_endpoint(client, sample_stage_images):
    """Verifies POST /api/v1/screen on verified clinical sample."""
    sample = sample_stage_images[2]
    files = {"file": (sample["filename"], sample["bytes"], "image/png")}
    response = client.post(
        "/api/v1/screen?use_tta=true&include_overlay=true&include_csme=true",
        files=files
    )
    assert response.status_code == 200
    data = response.json()

    assert data["quality"]["is_valid"] is True
    assert data["predicted_stage"] == 2
    assert "Stage 2" in data["stage_name"]
    assert data["referable_dr"] is True
    assert len(data["stage_probabilities"]) == 5
    assert "csme" in data
    assert data["csme"]["risk_level"] in ["Low", "Moderate", "High"]

    # Verify base64 overlays
    assert data["cam_overlay_base64"] is not None
    overlay_bytes = base64.b64decode(data["cam_overlay_base64"])
    assert len(overlay_bytes) > 500

    assert data["csme_annotated_base64"] is not None
    csme_bytes = base64.b64decode(data["csme_annotated_base64"])
    assert len(csme_bytes) > 500

def test_screen_corrupted_file(client):
    """Verifies that non-image file uploads return HTTP 400 Bad Request."""
    files = {"file": ("corrupt.png", b"this is not an image", "image/png")}
    response = client.post("/api/v1/screen", files=files)
    assert response.status_code == 400
    assert "not a valid or readable image" in response.json()["detail"]

def test_screen_bilateral_endpoint(client, sample_stage_images):
    """Verifies POST /api/v1/screen-bilateral with Right Eye (OD) and Left Eye (OS)."""
    sample_od = sample_stage_images[2]  # Stage 2
    sample_os = sample_stage_images[0]  # Stage 0

    files = {
        "file_od": (sample_od["filename"], sample_od["bytes"], "image/png"),
        "file_os": (sample_os["filename"], sample_os["bytes"], "image/png")
    }
    response = client.post("/api/v1/screen-bilateral?use_tta=true&include_overlay=true", files=files)
    assert response.status_code == 200
    data = response.json()

    assert data["od"]["predicted_stage"] == 2
    assert data["os"]["predicted_stage"] == 0
    assert data["overall_stage"] == 2
    assert "Right Eye (OD)" in data["governing_eye"]
    assert data["asymmetry_stage_delta"] == 2
    assert data["clinical_red_flag"] is True

def test_screen_batch_endpoint(client, sample_stage_images):
    """Verifies POST /api/v1/screen-batch triage on multiple images."""
    files = [
        ("files", (sample_stage_images[0]["filename"], sample_stage_images[0]["bytes"], "image/png")),
        ("files", (sample_stage_images[1]["filename"], sample_stage_images[1]["bytes"], "image/png")),
        ("files", (sample_stage_images[2]["filename"], sample_stage_images[2]["bytes"], "image/png"))
    ]
    response = client.post("/api/v1/screen-batch", files=files)
    assert response.status_code == 200
    data = response.json()

    assert data["total_scanned"] == 3
    assert data["quality_passed"] == 3
    assert len(data["manifest"]) == 3
    assert "RETINA INSIGHT CAMP SUMMARY" in data["executive_summary"]

def test_export_single_pdf_endpoint(client, sample_stage_images):
    """Verifies POST /api/v1/report/pdf streams valid PDF."""
    sample = sample_stage_images[2]
    files = {"file": (sample["filename"], sample["bytes"], "image/png")}
    data = {
        "patient_name": "Suresh Patel",
        "age": 58,
        "gender": "Male",
        "duration_years": 11,
        "hba1c": 8.7,
        "location": "Secunderabad"
    }
    response = client.post("/api/v1/report/pdf", files=files, data=data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")
    assert len(response.content) > 5000

def test_export_bilateral_pdf_endpoint(client, sample_stage_images):
    """Verifies POST /api/v1/report/bilateral-pdf streams valid bilateral PDF."""
    files = {
        "file_od": (sample_stage_images[2]["filename"], sample_stage_images[2]["bytes"], "image/png"),
        "file_os": (sample_stage_images[1]["filename"], sample_stage_images[1]["bytes"], "image/png")
    }
    data = {
        "patient_name": "Anita Sharma",
        "age": 52,
        "gender": "Female",
        "duration_years": 6,
        "hba1c": 7.4,
        "location": "Hyderabad"
    }
    response = client.post("/api/v1/report/bilateral-pdf", files=files, data=data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")
    assert len(response.content) > 5000

def test_export_camp_archive_zip_endpoint(client, sample_stage_images):
    """Verifies POST /api/v1/report/camp-archive streams valid ZIP package."""
    files = [
        ("files", (sample_stage_images[0]["filename"], sample_stage_images[0]["bytes"], "image/png")),
        ("files", (sample_stage_images[2]["filename"], sample_stage_images[2]["bytes"], "image/png"))
    ]
    response = client.post("/api/v1/report/camp-archive", files=files)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.content.startswith(b"PK")

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        names = z.namelist()
        assert "camp_triage_manifest.csv" in names
        assert "camp_executive_summary.txt" in names

def test_export_pdf_with_clinician_signoff_endpoint(client, sample_stage_images):
    """Verifies POST /api/v1/report/pdf with clinician accreditation form fields."""
    sample = sample_stage_images[1]
    files = {"file": (sample["filename"], sample["bytes"], "image/png")}
    data = {
        "patient_name": "Dr. Verified Patient",
        "age": 60,
        "gender": "Male",
        "duration_years": 10,
        "hba1c": 7.9,
        "location": "Hyderabad",
        "clinician_name": "Dr. A. Sharma, MD (Retina)",
        "clinician_reg_no": "MCI-RET-2018-9844"
    }
    response = client.post("/api/v1/report/pdf", files=files, data=data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")
    assert len(response.content) > 5000
