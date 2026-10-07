"""429 handling: waits are bounded, honour Retry-After, and end in a clear message."""
import json
from unittest.mock import patch

import httpx
import pytest

from tests.fake_zendesk import FakeZendesk, make_ticket, make_user, patch_zendesk
from zendesk_mcp import auth
from zendesk_mcp.config import DEFAULT_RATE_LIMIT_BUDGET_SECONDS, rate_limit_budget


@pytest.fixture(autouse=True)
def _default_budget(monkeypatch):
    monkeypatch.delenv("ZENDESK_MCP_RATELIMIT_BUDGET_SECONDS", raising=False)
    monkeypatch.setattr("zendesk_mcp.config.load_config", lambda *a, **k: {})


def _fake():
    return FakeZendesk([make_ticket(1, requester_id=1)], [make_user(1)])


# --- budget setting --------------------------------------------------------


def test_default_budget():
    assert rate_limit_budget(cfg={}, env={}) == DEFAULT_RATE_LIMIT_BUDGET_SECONDS == 20


def test_budget_from_config_and_env_with_env_winning():
    assert rate_limit_budget(cfg={"rate_limit_budget_seconds": 7}, env={}) == 7
    assert rate_limit_budget(cfg={"rate_limit_budget_seconds": 7},
                             env={"ZENDESK_MCP_RATELIMIT_BUDGET_SECONDS": "3"}) == 3


@pytest.mark.parametrize("bad", ["abc", "", "  "])
def test_invalid_budget_falls_back_to_default(bad):
    assert rate_limit_budget(cfg={}, env={"ZENDESK_MCP_RATELIMIT_BUDGET_SECONDS": bad}) == 20


def test_negative_budget_is_treated_as_zero():
    assert rate_limit_budget(cfg={}, env={"ZENDESK_MCP_RATELIMIT_BUDGET_SECONDS": "-5"}) == 0


# --- Zenpy path (tools that use get_client) --------------------------------


def test_zenpy_waits_out_a_short_429_and_succeeds(monkeypatch):
    fake = _fake()
    fake.queue_429(count=1, retry_after="2")
    patch_zendesk(monkeypatch, fake)
    from zendesk_mcp.tools.ticket import _get_ticket_data

    with patch("zenpy.lib.api.sleep") as sleep:
        result = json.loads(_get_ticket_data(1))

    assert result["id"] == 1
    assert sleep.call_count == 2  # Zenpy sleeps in one-second ticks


def test_zenpy_gives_up_with_a_clear_message_when_the_wait_budget_is_spent(monkeypatch):
    fake = _fake()
    fake.queue_429(count=1000, retry_after="30")
    patch_zendesk(monkeypatch, fake)
    monkeypatch.setenv("ZENDESK_MCP_RATELIMIT_BUDGET_SECONDS", "5")
    from zendesk_mcp.tools.ticket import _get_ticket_data

    with patch("zenpy.lib.api.sleep") as sleep:
        message = _get_ticket_data(1)

    assert message == auth.RATE_LIMIT_MESSAGE
    assert sleep.call_count <= 5  # never waits longer than the budget
    assert len(fake.calls) < 10   # and does not hammer the API either


def test_zero_budget_reports_the_429_without_waiting(monkeypatch):
    fake = _fake()
    fake.queue_429(count=1000, retry_after="30")
    patch_zendesk(monkeypatch, fake)
    monkeypatch.setenv("ZENDESK_MCP_RATELIMIT_BUDGET_SECONDS", "0")
    from zendesk_mcp.tools.ticket import _get_ticket_data

    with patch("zenpy.lib.api.sleep") as sleep:
        message = _get_ticket_data(1)

    assert message == auth.RATE_LIMIT_MESSAGE
    assert sleep.call_count == 0


def test_a_429_without_retry_after_is_reported_not_retried_forever(monkeypatch):
    fake = _fake()
    fake.queue_429(count=1000, retry_after=None)
    patch_zendesk(monkeypatch, fake)
    from zendesk_mcp.tools.ticket import _get_ticket_data

    with patch("zenpy.lib.api.sleep"):
        message = _get_ticket_data(1)

    assert message == auth.RATE_LIMIT_MESSAGE
    assert len(fake.calls) == 1


# --- direct HTTP path (auth.request) ---------------------------------------


def _resp(status, headers=None):
    return httpx.Response(status, json={}, headers=headers or {},
                          request=httpx.Request("GET", "https://acme.zendesk.com/x"))


@pytest.fixture
def cfg(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"subdomain": "acme", "oauth_token": "tok"}))
    return path


@patch("zendesk_mcp.auth.time.sleep")
@patch("zendesk_mcp.auth.httpx.request")
def test_request_waits_for_retry_after_then_succeeds(mock_request, sleep, cfg):
    mock_request.side_effect = [_resp(429, {"Retry-After": "2"}), _resp(200)]

    response = auth.request("GET", "https://acme.zendesk.com/x", config_file=cfg)

    assert response.status_code == 200
    sleep.assert_called_once_with(2)


@patch("zendesk_mcp.auth.time.sleep")
@patch("zendesk_mcp.auth.httpx.request")
def test_request_returns_the_429_when_retry_after_exceeds_the_budget(mock_request, sleep, cfg, monkeypatch):
    monkeypatch.setenv("ZENDESK_MCP_RATELIMIT_BUDGET_SECONDS", "5")
    mock_request.return_value = _resp(429, {"Retry-After": "60"})

    response = auth.request("GET", "https://acme.zendesk.com/x", config_file=cfg)

    assert response.status_code == 429
    sleep.assert_not_called()
    assert mock_request.call_count == 1


@patch("zendesk_mcp.auth.time.sleep")
@patch("zendesk_mcp.auth.httpx.request")
def test_request_backs_off_exponentially_without_retry_after_and_stops(mock_request, sleep, cfg):
    mock_request.return_value = _resp(429)

    response = auth.request("GET", "https://acme.zendesk.com/x", config_file=cfg)

    assert response.status_code == 429
    assert [c.args[0] for c in sleep.call_args_list] == [1, 2, 4]
    assert mock_request.call_count == auth.MAX_RATE_LIMIT_RETRIES + 1


@patch("zendesk_mcp.auth.time.sleep")
@patch("zendesk_mcp.auth.httpx.request")
def test_request_total_wait_never_exceeds_the_budget(mock_request, sleep, cfg, monkeypatch):
    monkeypatch.setenv("ZENDESK_MCP_RATELIMIT_BUDGET_SECONDS", "5")
    mock_request.return_value = _resp(429, {"Retry-After": "3"})

    auth.request("GET", "https://acme.zendesk.com/x", config_file=cfg)

    assert sum(c.args[0] for c in sleep.call_args_list) <= 5


@patch("zendesk_mcp.auth.time.sleep")
@patch("zendesk_mcp.auth.httpx.request")
def test_request_zero_budget_never_sleeps(mock_request, sleep, cfg, monkeypatch):
    monkeypatch.setenv("ZENDESK_MCP_RATELIMIT_BUDGET_SECONDS", "0")
    mock_request.return_value = _resp(429, {"Retry-After": "1"})

    assert auth.request("GET", "https://acme.zendesk.com/x", config_file=cfg).status_code == 429
    sleep.assert_not_called()


# --- error messages --------------------------------------------------------


def test_httpx_429_is_reported_as_a_rate_limit_not_a_generic_error():
    exc = httpx.HTTPStatusError("429", request=httpx.Request("GET", "https://x"), response=_resp(429))
    assert auth.api_error_message(exc) == auth.RATE_LIMIT_MESSAGE


def test_a_ticket_numbered_429_is_not_mistaken_for_a_rate_limit():
    assert auth.is_rate_limited(Exception("Ticket #429 not found")) is False
    assert auth.api_error_message(Exception("Ticket #429 not found")).startswith("Zendesk API error")
