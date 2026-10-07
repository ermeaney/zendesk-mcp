import json
import os
from pathlib import Path


def config_path() -> Path:
    return Path.home() / ".config" / "zendesk-mcp" / "config.json"


def load_config(path: Path | None = None) -> dict:
    resolved = path or config_path()
    try:
        return json.loads(resolved.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_config(data: dict, path: Path | None = None) -> None:
    resolved = path or config_path()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(data, indent=2) + "\n")
    resolved.chmod(0o600)


DEFAULT_RATE_LIMIT_BUDGET_SECONDS = 20


def rate_limit_budget(cfg: dict | None = None, env: dict | None = None) -> int:
    """Most seconds one tool call may spend waiting out Zendesk 429 responses.

    From ZENDESK_MCP_RATELIMIT_BUDGET_SECONDS, else ``rate_limit_budget_seconds`` in the
    config, else 20. 0 means never wait: a 429 is reported straight away.
    """
    cfg = load_config() if cfg is None else cfg
    env = os.environ if env is None else env
    for raw in (env.get("ZENDESK_MCP_RATELIMIT_BUDGET_SECONDS"), cfg.get("rate_limit_budget_seconds")):
        if raw is None or str(raw).strip() == "":
            continue
        try:
            return max(0, int(str(raw).strip()))
        except ValueError:
            continue
    return DEFAULT_RATE_LIMIT_BUDGET_SECONDS


def attachment_cache_dir(ticket_id: int, config_file: Path | None = None) -> Path:
    cfg = load_config(config_file)
    base = cfg.get("attachment_cache_dir", "~/.cache/zendesk-mcp/attachments")
    return Path(base).expanduser() / str(ticket_id)
