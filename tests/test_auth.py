import json
import subprocess
import sys
import textwrap
import threading
import time
from unittest.mock import patch

import httpx
import pytest

from zendesk_mcp import auth
from zendesk_mcp.auth import TokenExpiredError

NOW = 1_760_000_000

EXPIRED_401 = {
    "error": "invalid_token",
    "error_description": "The access token provided is expired, revoked, malformed or invalid for other reasons.",
}


def write_config(tmp_path, **overrides):
    cfg = {"subdomain": "acme", "oauth_token": "old-tok"}
    cfg.update(overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return path


def token_response(access="new-tok", refresh="new-refresh", expires_in=86400):
    body = {
        "access_token": access,
        "token_type": "bearer",
        "expires_in": expires_in,
        "refresh_token": refresh,
        "refresh_token_expires_in": 2592000,
        "scope": "read write",
    }
    return httpx.Response(
        status_code=200,
        json=body,
        request=httpx.Request("POST", "https://acme.zendesk.com/oauth/tokens"),
    )


# --- expiry detection ------------------------------------------------------


def test_is_expired_true_when_past():
    assert auth.is_expired({"expires_at": NOW - 1}, now=NOW) is True


def test_is_expired_false_when_far_future():
    assert auth.is_expired({"expires_at": NOW + 100_000}, now=NOW) is False


def test_is_expired_false_without_expires_at():
    """Legacy configs from a non-expiring OAuth client must not be treated as expired."""
    assert auth.is_expired({"oauth_token": "tok"}, now=NOW) is False


def test_is_expired_true_inside_skew_window():
    """Refresh slightly early so a call in flight does not land on a dead token."""
    assert auth.is_expired({"expires_at": NOW + 5}, now=NOW) is True


# --- valid_token -----------------------------------------------------------


@patch("zendesk_mcp.auth.httpx.post")
def test_valid_token_legacy_config_makes_no_network_call(mock_post, tmp_path):
    cfg = write_config(tmp_path)
    subdomain, token = auth.valid_token(cfg, now=NOW)
    assert (subdomain, token) == ("acme", "old-tok")
    mock_post.assert_not_called()


@patch("zendesk_mcp.auth.httpx.post")
def test_valid_token_unexpired_makes_no_network_call(mock_post, tmp_path):
    cfg = write_config(tmp_path, refresh_token="r", expires_at=NOW + 100_000)
    subdomain, token = auth.valid_token(cfg, now=NOW)
    assert token == "old-tok"
    mock_post.assert_not_called()


@patch("zendesk_mcp.auth.httpx.post")
def test_valid_token_refreshes_when_expired(mock_post, tmp_path):
    mock_post.return_value = token_response()
    cfg = write_config(
        tmp_path,
        refresh_token="the-refresh",
        expires_at=NOW - 10,
        client_id="cid",
        client_secret="secret",
    )

    subdomain, token = auth.valid_token(cfg, now=NOW)

    assert token == "new-tok"
    sent = mock_post.call_args.kwargs["json"]
    assert sent["grant_type"] == "refresh_token"
    assert sent["refresh_token"] == "the-refresh"
    assert sent["client_id"] == "cid"
    assert sent["client_secret"] == "secret"
    assert "acme.zendesk.com/oauth/tokens" in mock_post.call_args.args[0]


@patch("zendesk_mcp.auth.httpx.post")
def test_refresh_persists_rotated_credentials(mock_post, tmp_path):
    """Zendesk invalidates the old refresh token on use, so the new one must be saved."""
    mock_post.return_value = token_response(access="a2", refresh="r2", expires_in=3600)
    cfg = write_config(
        tmp_path, refresh_token="r1", expires_at=NOW - 10, client_id="cid", client_secret="s"
    )

    auth.valid_token(cfg, now=NOW)

    saved = json.loads(cfg.read_text())
    assert saved["oauth_token"] == "a2"
    assert saved["refresh_token"] == "r2"
    assert saved["expires_at"] == NOW + 3600
    assert saved["subdomain"] == "acme"


@patch("zendesk_mcp.auth.httpx.post")
def test_refresh_preserves_unrelated_config_keys(mock_post, tmp_path):
    mock_post.return_value = token_response()
    cfg = write_config(
        tmp_path,
        refresh_token="r1",
        expires_at=NOW - 10,
        client_id="cid",
        git_zen_field_id=4321,
        attachment_cache_dir="~/.cache/zendesk-mcp/attachments",
    )

    auth.valid_token(cfg, now=NOW)

    saved = json.loads(cfg.read_text())
    assert saved["git_zen_field_id"] == 4321
    assert saved["attachment_cache_dir"] == "~/.cache/zendesk-mcp/attachments"


def test_valid_token_expired_without_refresh_token_tells_user_to_rerun_setup(tmp_path):
    """The currently-broken population: token expires but nothing on disk can renew it."""
    cfg = write_config(tmp_path, expires_at=NOW - 10)
    with pytest.raises(TokenExpiredError) as exc:
        auth.valid_token(cfg, now=NOW)
    assert "zendesk-mcp setup" in str(exc.value)


@patch("zendesk_mcp.auth.httpx.post")
def test_valid_token_raises_when_refresh_token_rejected(mock_post, tmp_path):
    """Refresh token expired (30d) or revoked -> user must re-authorize."""
    mock_post.return_value = httpx.Response(
        status_code=400,
        json={"error": "invalid_grant", "error_description": "The refresh token is invalid."},
        request=httpx.Request("POST", "https://acme.zendesk.com/oauth/tokens"),
    )
    cfg = write_config(
        tmp_path, refresh_token="dead", expires_at=NOW - 10, client_id="cid", client_secret="s"
    )

    with pytest.raises(TokenExpiredError) as exc:
        auth.valid_token(cfg, now=NOW)
    assert "zendesk-mcp setup" in str(exc.value)


@patch("zendesk_mcp.auth.httpx.post")
def test_refresh_does_not_clobber_a_token_another_process_already_renewed(mock_post, tmp_path):
    """Two MCP servers share one config file; re-read before refreshing to avoid a rotation race."""
    cfg = write_config(tmp_path, refresh_token="r1", expires_at=NOW - 10, client_id="cid")
    stale = json.loads(cfg.read_text())
    # Another process refreshed after we loaded our copy.
    cfg.write_text(json.dumps({**stale, "oauth_token": "fresh-from-other", "expires_at": NOW + 5000}))

    subdomain, token = auth.valid_token(cfg, now=NOW, cfg_snapshot=stale)

    assert token == "fresh-from-other"
    mock_post.assert_not_called()


# --- error classification --------------------------------------------------


def test_api_error_message_upgrades_httpx_401():
    exc = httpx.HTTPStatusError(
        "Client error '401 Unauthorized' for url 'https://acme.zendesk.com/api/v2/users/search.json'",
        request=httpx.Request("GET", "https://acme.zendesk.com/api/v2/users/search.json"),
        response=httpx.Response(401, json=EXPIRED_401),
    )
    msg = auth.api_error_message(exc)
    assert "zendesk-mcp setup" in msg
    assert "expired" in msg.lower()


def test_api_error_message_upgrades_zenpy_invalid_token():
    from zenpy.lib.exception import APIException

    msg = auth.api_error_message(APIException(json.dumps(EXPIRED_401)))
    assert "zendesk-mcp setup" in msg


def test_api_error_message_leaves_other_errors_alone():
    msg = auth.api_error_message(Exception("404 Not Found"))
    assert msg == "Zendesk API error: 404 Not Found"
    assert "zendesk-mcp setup" not in msg


def test_api_error_message_does_not_flag_rate_limit():
    msg = auth.api_error_message(Exception("429 Too Many Requests"))
    assert "zendesk-mcp setup" not in msg


def test_api_error_message_does_not_flag_a_ticket_numbered_401():
    msg = auth.api_error_message(Exception("Ticket #401 not found"))
    assert "zendesk-mcp setup" not in msg


def test_api_error_message_does_not_flag_a_status_code_containing_401():
    msg = auth.api_error_message(Exception("upstream returned 4013"))
    assert "zendesk-mcp setup" not in msg


def test_api_error_message_flags_unauthorized_wording():
    msg = auth.api_error_message(Exception("Client error '401 Unauthorized' for url ..."))
    assert "zendesk-mcp setup" in msg


# --- reactive refresh on 401 ----------------------------------------------


@patch("zendesk_mcp.auth.httpx.request")
@patch("zendesk_mcp.auth.httpx.post")
def test_request_refreshes_and_retries_once_on_401(mock_post, mock_request, tmp_path):
    """Server-side revocation / clock skew: a 401 should trigger one refresh + retry."""
    mock_post.return_value = token_response(access="tok2")
    ok = httpx.Response(
        200, json={"users": []}, request=httpx.Request("GET", "https://acme.zendesk.com/x")
    )
    unauthorized = httpx.Response(
        401, json=EXPIRED_401, request=httpx.Request("GET", "https://acme.zendesk.com/x")
    )
    mock_request.side_effect = [unauthorized, ok]
    cfg = write_config(
        tmp_path,
        refresh_token="r1",
        expires_at=NOW + 100_000,  # looks valid, but server disagrees
        client_id="cid",
        client_secret="s",
    )

    response = auth.request("GET", "https://acme.zendesk.com/x", config_file=cfg, now=NOW)

    assert response.status_code == 200
    assert mock_request.call_count == 2
    mock_post.assert_called_once()
    # retry must carry the refreshed bearer token
    assert mock_request.call_args_list[1].kwargs["headers"]["Authorization"] == "Bearer tok2"


@patch("zendesk_mcp.auth.httpx.request")
@patch("zendesk_mcp.auth.httpx.post")
def test_request_does_not_retry_more_than_once(mock_post, mock_request, tmp_path):
    mock_post.return_value = token_response()
    unauthorized = httpx.Response(
        401, json=EXPIRED_401, request=httpx.Request("GET", "https://acme.zendesk.com/x")
    )
    mock_request.side_effect = [unauthorized, unauthorized]
    cfg = write_config(
        tmp_path, refresh_token="r1", expires_at=NOW + 100_000, client_id="cid", client_secret="s"
    )

    with pytest.raises(TokenExpiredError):
        auth.request("GET", "https://acme.zendesk.com/x", config_file=cfg, now=NOW)

    assert mock_request.call_count == 2


@patch("zendesk_mcp.auth.httpx.request")
def test_request_injects_bearer_token(mock_request, tmp_path):
    mock_request.return_value = httpx.Response(
        200, json={}, request=httpx.Request("GET", "https://acme.zendesk.com/x")
    )
    cfg = write_config(tmp_path)

    auth.request("GET", "https://acme.zendesk.com/x", config_file=cfg, now=NOW)

    assert mock_request.call_args.kwargs["headers"]["Authorization"] == "Bearer old-tok"


# --- cross-process refresh lock --------------------------------------------


def test_refresh_lock_file_sits_next_to_the_config(tmp_path):
    cfg = write_config(tmp_path)
    with auth._refresh_lock(cfg):
        assert (tmp_path / "config.json.lock").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl locking is POSIX only")
def test_refresh_lock_blocks_another_process_until_released(tmp_path):
    cfg = write_config(tmp_path)
    holder = subprocess.Popen(
        [sys.executable, "-I", "-c", textwrap.dedent(f"""
            import fcntl, os, sys, time
            fd = os.open({str(tmp_path / "config.json.lock")!r}, os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            print("locked", flush=True)
            time.sleep(0.8)
        """)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "locked"
        started = time.monotonic()
        with auth._refresh_lock(cfg):
            waited = time.monotonic() - started
    finally:
        holder.wait(timeout=10)
    assert waited >= 0.4


@patch("zendesk_mcp.auth.httpx.post")
def test_concurrent_refreshes_redeem_the_refresh_token_only_once(mock_post, tmp_path):
    """Several MCP server processes see the same expired token at once; only one may refresh."""
    cfg = write_config(tmp_path, refresh_token="r1", expires_at=NOW - 10, client_id="cid")
    stale = json.loads(cfg.read_text())
    calls = []

    def slow_refresh(*_args, **_kwargs):
        calls.append(1)
        time.sleep(0.2)  # long enough for the other callers to queue on the lock
        return token_response(access=f"tok-{len(calls)}", refresh=f"refresh-{len(calls)}", expires_in=86400)

    mock_post.side_effect = slow_refresh
    results, errors = [], []

    def worker():
        try:
            # Every caller holds the same pre-refresh snapshot, like separate processes would.
            results.append(auth.valid_token(cfg, now=NOW, cfg_snapshot=dict(stale)))
        except Exception as e:  # pragma: no cover - failure path
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=30)

    assert errors == []
    assert len(calls) == 1
    assert {token for _, token in results} == {"tok-1"}
    assert json.loads(cfg.read_text())["refresh_token"] == "refresh-1"


@patch("zendesk_mcp.auth.httpx.request")
@patch("zendesk_mcp.auth.httpx.post")
def test_request_401_uses_credentials_another_process_just_stored(mock_post, mock_request, tmp_path):
    """A 401 must not redeem the refresh token if another process already replaced the access token."""
    cfg = write_config(
        tmp_path, refresh_token="r1", expires_at=NOW + 100_000, client_id="cid", client_secret="s"
    )
    ok = httpx.Response(200, json={}, request=httpx.Request("GET", "https://acme.zendesk.com/x"))
    unauthorized = httpx.Response(
        401, json=EXPIRED_401, request=httpx.Request("GET", "https://acme.zendesk.com/x")
    )

    def first_call_loses_the_race(*_args, **_kwargs):
        if mock_request.call_count == 1:
            stored = json.loads(cfg.read_text())
            cfg.write_text(json.dumps({**stored, "oauth_token": "other-process-tok", "refresh_token": "r2"}))
            return unauthorized
        return ok

    mock_request.side_effect = first_call_loses_the_race

    response = auth.request("GET", "https://acme.zendesk.com/x", config_file=cfg, now=NOW)

    assert response.status_code == 200
    mock_post.assert_not_called()
    assert mock_request.call_args_list[1].kwargs["headers"]["Authorization"] == "Bearer other-process-tok"
