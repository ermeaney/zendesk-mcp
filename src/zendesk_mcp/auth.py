"""OAuth access-token lifecycle for the Zendesk API.

Zendesk applies a default expiry to access tokens issued by OAuth clients created on or
after 2026-04-30 (30 minutes, with a 30-day refresh token). Clients created earlier still
issue non-expiring tokens unless ``expires_in`` is requested explicitly. Both shapes have
to keep working, so a config without ``expires_at`` is treated as non-expiring and is
never refreshed.

Refreshing rotates the refresh token and invalidates the previous pair, so the new
credentials must be persisted on every refresh. Several server processes (one per MCP
client session) share one config file, so a refresh runs under an exclusive file lock and
re-reads the file once it holds it: a process that waited uses the credentials the first one
stored instead of redeeming the refresh token a second time, which would be rejected and
force the user to re-run setup. The lock needs ``fcntl``, so on Windows only the re-read
protects against the race.
"""
import os
import re
import time
from contextlib import contextmanager
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx
from zenpy.lib.exception import RatelimitBudgetExceeded

from zendesk_mcp.config import config_path, load_config, rate_limit_budget, save_config

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    fcntl = None

# Refresh slightly ahead of the deadline so a request already in flight does not
# land on a token that expired in transit.
REFRESH_SKEW_SECONDS = 120
REFRESH_TIMEOUT_SECONDS = 30

REAUTH_HINT = "Re-run: zendesk-mcp setup"

# 429 handling for the direct HTTP calls (Zenpy has its own, bounded by ratelimit_budget).
MAX_RATE_LIMIT_RETRIES = 3
RATE_LIMIT_MESSAGE = (
    "Zendesk is rate limiting this account (HTTP 429), and the request was still being "
    "throttled after waiting. Other integrations share the same API limit. Wait a minute "
    "and try again."
)


class TokenExpiredError(Exception):
    """The access token is unusable and could not be renewed automatically."""


def _expired_message(detail: str = "") -> str:
    base = (
        "Zendesk authorization failed: the OAuth access token is expired, revoked, or "
        "invalid, and could not be refreshed automatically."
    )
    if detail:
        base = f"{base} ({detail})"
    return f"{base} {REAUTH_HINT}"


def is_expired(cfg: dict, now: float | None = None) -> bool:
    """True if the stored token has passed (or is about to pass) its expiry.

    A config with no ``expires_at`` came from a non-expiring OAuth client and is
    reported as valid.
    """
    expires_at = cfg.get("expires_at")
    if not expires_at:
        return False
    return (now if now is not None else time.time()) >= expires_at - REFRESH_SKEW_SECONDS


def can_refresh(cfg: dict) -> bool:
    return bool(cfg.get("refresh_token") and cfg.get("client_id"))


def refresh_access_token(
    cfg: dict, config_file: Path | None = None, now: float | None = None
) -> dict:
    """Exchange the stored refresh token for a new token pair and persist it.

    Returns the updated config. Raises TokenExpiredError if the refresh is rejected.
    """
    if not can_refresh(cfg):
        raise TokenExpiredError(_expired_message("no refresh token stored"))

    now = now if now is not None else time.time()
    subdomain = cfg.get("subdomain", "").strip()
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": cfg["refresh_token"],
        "client_id": cfg["client_id"],
    }
    if cfg.get("client_secret"):
        payload["client_secret"] = cfg["client_secret"]

    try:
        response = httpx.post(
            f"https://{subdomain}.zendesk.com/oauth/tokens",
            json=payload,
            timeout=REFRESH_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = response.json()
    except httpx.HTTPStatusError as e:
        # invalid_grant => the refresh token itself is expired (30d default) or revoked.
        raise TokenExpiredError(
            _expired_message(f"refresh rejected with HTTP {e.response.status_code}")
        ) from e
    except Exception as e:
        raise TokenExpiredError(_expired_message(f"refresh failed: {e}")) from e

    access_token = body.get("access_token")
    if not access_token:
        raise TokenExpiredError(_expired_message("refresh response had no access_token"))

    updated = dict(cfg)
    updated["oauth_token"] = access_token
    # Zendesk invalidates the old refresh token on use; keep the rotated one.
    if body.get("refresh_token"):
        updated["refresh_token"] = body["refresh_token"]
    if body.get("expires_in"):
        updated["expires_at"] = int(now + body["expires_in"])
    save_config(updated, config_file)
    return updated


@contextmanager
def _refresh_lock(config_file: Path | None = None):
    """Exclusive cross-process lock held while redeeming a refresh token."""
    if fcntl is None:  # pragma: no cover - Windows
        yield
        return
    target = config_file or config_path()
    lock_path = target.with_name(target.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)  # closing the descriptor releases the lock


def _renewed_credentials(
    used_token: str,
    config_file: Path | None,
    now: float | None,
    fallback: dict | None = None,
) -> dict:
    """Return credentials that replace ``used_token``, refreshing at most once across processes.

    ``used_token`` is the access token the caller found unusable. If the file now holds a
    different, unexpired token, another process already refreshed while we waited for the
    lock, and redeeming the refresh token again would invalidate that fresh pair.
    """
    with _refresh_lock(config_file):
        on_disk = load_config(config_file)
        current = (on_disk.get("oauth_token") or "").strip()
        if current and current != used_token and not is_expired(on_disk, now):
            return on_disk
        return refresh_access_token(on_disk or fallback or {}, config_file, now)


def valid_token(
    config_file: Path | None = None,
    now: float | None = None,
    cfg_snapshot: dict | None = None,
) -> tuple[str, str]:
    """Return (subdomain, access_token), refreshing first if the token has expired.

    Raises TokenExpiredError when the token is expired and cannot be renewed.
    """
    cfg = cfg_snapshot if cfg_snapshot is not None else load_config(config_file)
    subdomain = cfg.get("subdomain", "").strip()
    token = cfg.get("oauth_token", "").strip()
    if not subdomain or not token:
        # Caller maps this to the existing "not configured" ConfigError.
        return subdomain, token

    if not is_expired(cfg, now):
        return subdomain, token

    cfg = _renewed_credentials(token, config_file, now, fallback=cfg)
    return cfg.get("subdomain", "").strip(), cfg["oauth_token"].strip()


def _retry_after_seconds(response: httpx.Response, attempt: int) -> int:
    """Seconds Zendesk asked us to wait, or an exponential backoff if it did not say."""
    raw = response.headers.get("Retry-After", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            try:
                delta = parsedate_to_datetime(raw).timestamp() - time.time()
                return max(1, int(delta))
            except (TypeError, ValueError):
                pass
    return 2 ** attempt


def _send(method: str, url: str, headers: dict, budget: list, **kwargs) -> httpx.Response:
    """httpx.request that waits out 429 responses, within ``budget`` seconds in total.

    ``budget`` is a one-element list so the wait budget is shared with the caller's other
    attempts. A 429 means the request was not processed, so retrying is safe for any method.
    If the wait Zendesk asked for does not fit in what is left, the 429 is returned as is.
    """
    attempt = 0
    while True:
        response = httpx.request(method, url, headers=headers, **kwargs)
        if response.status_code != 429 or attempt >= MAX_RATE_LIMIT_RETRIES:
            return response
        wait = _retry_after_seconds(response, attempt)
        if wait > budget[0]:
            return response
        budget[0] -= wait
        time.sleep(wait)
        attempt += 1


def request(
    method: str,
    url: str,
    config_file: Path | None = None,
    now: float | None = None,
    **kwargs,
) -> httpx.Response:
    """httpx.request with the bearer token injected, refreshing once on a 401.

    Pre-emptive expiry checking covers the normal case; this also handles a token
    revoked server-side or a clock skew that made it look valid. 429 responses are waited
    out within the rate-limit budget (see config.rate_limit_budget).
    """
    kwargs.setdefault("timeout", 30)
    _, token = valid_token(config_file, now)
    headers = {**kwargs.pop("headers", {}), "Authorization": f"Bearer {token}"}
    budget = [rate_limit_budget()]
    response = _send(method, url, headers, budget, **kwargs)

    if response.status_code != 401:
        return response

    cfg = load_config(config_file)
    if not can_refresh(cfg):
        raise TokenExpiredError(_expired_message("server rejected the token"))

    cfg = _renewed_credentials(token, config_file, now, fallback=cfg)
    headers["Authorization"] = f"Bearer {cfg['oauth_token']}"
    response = _send(method, url, headers, budget, **kwargs)
    if response.status_code == 401:
        raise TokenExpiredError(_expired_message("still rejected after refresh"))
    return response


_AUTH_KEYWORDS = ("invalid_token", "invalid_grant", "unauthorized", "couldn't authenticate")


def is_auth_error(exc: Exception) -> bool:
    """True if an exception from a Zendesk call means the credentials were rejected."""
    if isinstance(exc, TokenExpiredError):
        return True
    # Structured signals first: an httpx HTTPStatusError or a zenpy APIException both
    # carry the originating response.
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == 401:
        return True

    text = str(exc).lower()
    if any(keyword in text for keyword in _AUTH_KEYWORDS):
        return True
    # Fall back to the status code in a stringified error, but keep it word-bounded so
    # "4013" or a ticket numbered 401 is not mistaken for an auth failure.
    if "not found" in text:
        return False
    return re.search(r"(?<!\d)401(?!\d)", text) is not None


def is_rate_limited(exc: Exception) -> bool:
    """True if an exception from a Zendesk call means the API throttled us (HTTP 429)."""
    if isinstance(exc, RatelimitBudgetExceeded):
        return True
    if getattr(getattr(exc, "response", None), "status_code", None) == 429:
        return True
    text = str(exc).lower()
    return "too many requests" in text or "rate limit" in text or "ratelimit" in text


def api_error_message(exc: Exception) -> str:
    """Render an exception from a Zendesk call as a message for the caller.

    Authorization failures get remediation; everything else keeps the existing format.
    """
    if isinstance(exc, TokenExpiredError):
        return str(exc)
    if is_rate_limited(exc):
        return RATE_LIMIT_MESSAGE
    if is_auth_error(exc):
        return _expired_message()
    return f"Zendesk API error: {exc}"
