"""Regression checks for Docker deployment artifacts (no docker daemon required)."""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

REPO = Path(__file__).resolve().parents[1]


def test_docker_compose_wd_tagger_hydrus_networking():
    text = (REPO / "docker-compose.yml").read_text(encoding="utf-8")
    for needle in (
        "HYDRUS_API_URL=${HYDRUS_API_URL:-http://host.docker.internal:45869}",
        "host.docker.internal:host-gateway",
        "8199:8199",
        'profiles: ["hydrus-web"]',
        "HYDRUS_WEB_URL",
        "./config.yaml:/app/config.yaml:ro",
        "./face_data:/app/face_data",
    ):
        assert needle in text, f"expected {needle!r} in docker-compose.yml"


def test_dockerfile_non_root_healthcheck():
    text = (REPO / "Dockerfile").read_text(encoding="utf-8")
    for needle in (
        "USER tagger",
        "HEALTHCHECK",
        "/api/app/status",
        "python:3.11-slim",
    ):
        assert needle in text, f"expected {needle!r} in Dockerfile"


def test_dockerfile_face_deps_and_libs():
    text = (REPO / "Dockerfile").read_text(encoding="utf-8")
    for needle in (
        '".[face]"',
        "libxcb1",
        "opencv-python",
    ):
        assert needle in text, f"expected {needle!r} in Dockerfile"


def test_dockerignore_excludes_secrets_and_tests():
    text = (REPO / ".dockerignore").read_text(encoding="utf-8")
    for needle in ("config.yaml", "tests/", ".venv/"):
        assert needle in text, f"expected {needle!r} in .dockerignore"


def test_docker_smoke_script_probes_tagger_and_hydrus_web():
    path = REPO / "scripts" / "docker_smoke.sh"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "/api/app/status" in text
    assert "HYDRUS_WEB_URL" in text
