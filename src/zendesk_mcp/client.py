from pathlib import Path
from zenpy import Zenpy
from zendesk_mcp.auth import valid_token
from zendesk_mcp.config import rate_limit_budget


class ConfigError(Exception):
    pass


def get_client(config_file: Path | None = None) -> Zenpy:
    subdomain, token = valid_token(config_file)
    if not subdomain or not token:
        raise ConfigError("Zendesk not configured. Run: zendesk-mcp setup")
    # Zenpy waits out 429 responses, with no limit unless a budget is set. Cap it so a tool call
    # fails with a clear message instead of blocking the agent for as long as Zendesk asks.
    return Zenpy(subdomain=subdomain, oauth_token=token, ratelimit_budget=rate_limit_budget())


def get_oauth_session() -> tuple[str, str]:
    """Return (subdomain, oauth_token) for direct API calls, refreshing an expired token.

    Raises ConfigError if unconfigured, TokenExpiredError if the token cannot be renewed.
    """
    subdomain, token = valid_token()
    if not subdomain or not token:
        raise ConfigError("Zendesk not configured. Run: zendesk-mcp setup")
    return subdomain, token
