"""Hydrus connectivity error classification."""

import httpx
import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.hydrus.transport_errors import is_hydrus_transport_error


def test_transport_errors_map_httpx_network_types():
    assert is_hydrus_transport_error(httpx.ConnectError("x")) is True
    assert is_hydrus_transport_error(httpx.ReadError("x")) is True
    assert is_hydrus_transport_error(httpx.TimeoutException("x")) is True


@pytest.mark.parametrize(
    "status,expected",
    [(502, True), (503, True), (408, True), (500, False), (404, False)],
)
def test_http_status_error_by_code(status, expected):
    req = httpx.Request("GET", "http://example.test")
    resp = httpx.Response(status, request=req)
    exc = httpx.HTTPStatusError("msg", request=req, response=resp)
    assert is_hydrus_transport_error(exc) is expected


def test_http_status_error_missing_response_is_not_transport():
    """``HTTPStatusError`` with no response must not be treated as a transport failure."""
    req = httpx.Request("GET", "http://example.test")
    resp = httpx.Response(502, request=req)
    exc = httpx.HTTPStatusError("msg", request=req, response=resp)
    exc.response = None
    assert is_hydrus_transport_error(exc) is False


def test_http_status_error_unknown_exception_false():
    assert is_hydrus_transport_error(ValueError("x")) is False
