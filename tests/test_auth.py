"""The /ask auth check must fail closed: no configured key means no access."""
import pytest
from fastapi import HTTPException

from app.api.main import require_api_key
from app.config import settings


def test_no_server_key_rejects_with_503(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "")
    with pytest.raises(HTTPException) as exc:
        require_api_key("")  # empty header vs empty key: the compare_digest trap
    assert exc.value.status_code == 503


def test_wrong_key_rejects_with_401(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "correct-key")
    with pytest.raises(HTTPException) as exc:
        require_api_key("wrong-key")
    assert exc.value.status_code == 401


def test_missing_header_rejects_with_401(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "correct-key")
    with pytest.raises(HTTPException) as exc:
        require_api_key("")
    assert exc.value.status_code == 401


def test_correct_key_is_accepted(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "correct-key")
    require_api_key("correct-key")  # no exception = access granted