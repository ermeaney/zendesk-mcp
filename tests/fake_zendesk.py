"""A fake Zendesk API for tests, plugged in as a requests transport adapter.

It serves canned JSON for the endpoints the server uses, records every request, and can
inject 429 responses. Nothing touches the network, so tests can assert how many API calls a
tool makes and how it behaves under rate limiting.
"""
import json
from urllib.parse import parse_qs, urlparse

import requests
from requests.adapters import BaseAdapter

TS = "2026-01-01T00:00:00Z"


def make_user(user_id, name=None, role="end-user"):
    return {
        "id": user_id, "name": name or f"User {user_id}", "email": f"u{user_id}@example.com",
        "role": role, "created_at": TS, "updated_at": TS,
    }


def make_ticket(ticket_id, requester_id=1, assignee_id=None, group_id=None):
    return {
        "id": ticket_id, "subject": f"Ticket {ticket_id}", "description": "something broke",
        "status": "open", "priority": "normal", "type": "incident", "tags": ["a"],
        "requester_id": requester_id, "assignee_id": assignee_id, "group_id": group_id,
        "created_at": TS, "updated_at": TS,
        "url": f"https://acme.zendesk.com/api/v2/tickets/{ticket_id}.json",
    }


class FakeZendesk(BaseAdapter):
    def __init__(self, tickets=(), users=(), groups=(), comments=None):
        super().__init__()
        self.tickets = {t["id"]: t for t in tickets}
        self.users = {u["id"]: u for u in users}
        self.groups = {g["id"]: g for g in groups}
        self.comments = comments or {}
        self.calls: list[tuple[str, str]] = []
        self.throttle: list[tuple[int, dict]] = []  # queued (status, headers) overrides

    # --- test helpers -----------------------------------------------------

    def queue_429(self, count=1, retry_after="1"):
        headers = {"Retry-After": retry_after} if retry_after is not None else {}
        self.throttle.extend([(429, headers)] * count)

    def paths(self):
        return [p for _, p in self.calls]

    def session(self):
        session = requests.Session()
        session.mount("https://", self)
        return session

    # --- transport --------------------------------------------------------

    def close(self):
        pass

    def _respond(self, request, status, payload=None, headers=None):
        response = requests.Response()
        response.status_code = status
        response.headers.update(headers or {})
        response.headers.setdefault("Content-Type", "application/json")
        response._content = json.dumps(payload if payload is not None else {}).encode()
        response.url = request.url
        response.request = request
        return response

    def send(self, request, **kwargs):
        parsed = urlparse(request.url)
        path, query = parsed.path, parse_qs(parsed.query)
        self.calls.append((request.method, path + (f"?{parsed.query}" if parsed.query else "")))

        if self.throttle:
            status, headers = self.throttle.pop(0)
            return self._respond(request, status, {"error": "rate limited"}, headers)

        parts = [p for p in path.split("/") if p]  # ["api", "v2", ...]
        rest = parts[2:]
        if request.method == "GET":
            if rest[:1] == ["tickets"] and len(rest) == 2:
                tid = int(rest[1].removesuffix(".json"))
                if tid in self.tickets:
                    return self._respond(request, 200, {"ticket": self.tickets[tid]})
            if rest[:1] == ["tickets"] and len(rest) == 3 and rest[2] == "comments.json":
                tid = int(rest[1])
                items = self.comments.get(tid, [])
                return self._respond(request, 200, {"comments": items, "count": len(items), "next_page": None})
            if rest[:1] == ["users"] and len(rest) == 2 and rest[1] != "show_many.json":
                uid = int(rest[1].removesuffix(".json"))
                if uid in self.users:
                    return self._respond(request, 200, {"user": self.users[uid]})
            if rest == ["users", "show_many.json"]:
                ids = [int(i) for i in query.get("ids", [""])[0].split(",") if i]
                return self._respond(request, 200, {"users": [self.users[i] for i in ids if i in self.users], "count": len(ids), "next_page": None})
            if rest[:1] == ["groups"] and len(rest) == 2:
                gid = int(rest[1].removesuffix(".json"))
                if gid in self.groups:
                    return self._respond(request, 200, {"group": self.groups[gid]})
            if rest == ["search.json"]:
                per_page = 100
                page = int(query.get("page", ["1"])[0])
                rows = [dict(t, result_type="ticket") for t in self.tickets.values()]
                window = rows[(page - 1) * per_page: page * per_page]
                nxt = None
                if page * per_page < len(rows):
                    nxt = f"https://acme.zendesk.com/api/v2/search.json?page={page + 1}&{parsed.query}"
                return self._respond(request, 200, {"results": window, "count": len(rows), "next_page": nxt})
        return self._respond(request, 404, {"error": "RecordNotFound", "description": "Not found"})


def patch_zendesk(monkeypatch, fake):
    """Route get_client() through the fake, with a fixed token and no config file."""
    from zenpy import Zenpy
    import zendesk_mcp.client as client_mod

    monkeypatch.setattr(client_mod, "valid_token", lambda *a, **k: ("acme", "tok"))
    monkeypatch.setattr(client_mod, "Zenpy", lambda **kw: Zenpy(session=fake.session(), **kw))
