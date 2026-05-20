"""
Tests for LSTM Stock API
Run with: pytest tests/ -v
"""
import json
import pytest
from fastapi.testclient import TestClient

# We test without a loaded model to validate routes and schema
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app

client = TestClient(app)


# ─────────────────────────────────────────────
# HEALTH & ROOT
# ─────────────────────────────────────────────
def test_root_returns_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "LSTM" in r.text


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "uptime_requests" in data


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "total_requests" in data
    assert "response_time_ms" in data
    assert "avg" in data["response_time_ms"]


# ─────────────────────────────────────────────
# PREDICT — model not loaded
# ─────────────────────────────────────────────
def test_predict_without_model_returns_503():
    payload = {"symbol": "PETR4.SA", "days_ahead": 3, "use_latest": False,
               "prices": [30.0] * 60}
    r = client.post("/predict", json=payload)
    assert r.status_code == 503


def test_predict_invalid_days_returns_422():
    payload = {"symbol": "PETR4.SA", "days_ahead": 99, "use_latest": False,
               "prices": [30.0] * 60}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_days_below_min_returns_422():
    payload = {"symbol": "PETR4.SA", "days_ahead": 0, "use_latest": False,
               "prices": [30.0] * 60}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


# ─────────────────────────────────────────────
# MODEL INFO — no model
# ─────────────────────────────────────────────
def test_model_info_without_model_returns_503():
    r = client.get("/model/info")
    assert r.status_code == 503


# ─────────────────────────────────────────────
# REQUEST TIMING MIDDLEWARE
# ─────────────────────────────────────────────
def test_response_time_header_present():
    r = client.get("/health")
    assert "x-response-time-ms" in r.headers


# ─────────────────────────────────────────────
# PREDICT SCHEMA VALIDATION
# ─────────────────────────────────────────────
def test_predict_without_prices_and_use_latest_false_returns_error():
    """If use_latest=False but no prices provided, should fail."""
    payload = {"days_ahead": 5, "use_latest": False}
    r = client.post("/predict", json=payload)
    # Either 503 (no model) or 422 (validation) — both acceptable without model
    assert r.status_code in (422, 503)
