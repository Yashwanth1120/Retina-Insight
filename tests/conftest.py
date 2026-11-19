"""
tests/conftest.py: Shared PyTest fixtures for Retina Insight test suite.
"""
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import pytest
from fastapi.testclient import TestClient
from api import app, SYSTEM_STATE
from engine import load_inference_system, get_sample_image

@pytest.fixture(scope="session")
def client():
    """Initializes FastAPI TestClient with loaded model lifecycle."""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture(scope="session")
def loaded_system():
    """Ensures model and calibration factors are loaded once for all unit tests."""
    model, grad_cam, temp, temp_from_file, device = load_inference_system(
        model_path="best_retina_model.pth",
        temp_path="temperature.txt"
    )
    return {
        "model": model,
        "grad_cam": grad_cam,
        "temperature": temp,
        "device": device
    }

@pytest.fixture(scope="session")
def sample_stage_images():
    """Loads 1 verified sample image path and byte payload for each of Stages 0-4."""
    samples = {}
    for stage in range(5):
        img_bgr, path = get_sample_image(stage, case_idx=0)
        assert img_bgr is not None, f"Sample image for stage {stage} could not be loaded."
        is_ok, buffer = cv2.imencode(".png", img_bgr)
        assert is_ok, f"Failed to encode sample image for stage {stage}"
        samples[stage] = {
            "path": path,
            "bgr": img_bgr,
            "bytes": buffer.tobytes(),
            "filename": os.path.basename(path)
        }
    return samples

@pytest.fixture
def mock_patient_info():
    """Returns standardized clinical patient metadata."""
    return {
        "name": "Ramesh Kumar",
        "age": 54,
        "gender": "Male",
        "duration": 9,
        "hba1c": 8.4,
        "location": "Hyderabad"
    }
