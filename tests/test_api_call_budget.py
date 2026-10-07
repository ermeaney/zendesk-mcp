"""Guards the number of Zendesk API calls each tool makes.

Zendesk's rate limit is shared by every API client on the account (Enterprise Plus is
2,500 requests per minute), and Zenpy resolves related records lazily, one request each. These
tests run the real tools against a fake Zendesk that records requests, so a change that turns a
single call into one-per-record fails here instead of during an incident.
"""
import pytest

from tests.fake_zendesk import FakeZendesk, make_ticket, make_user, patch_zendesk


def _account(n_tickets=50, n_assignees=5, n_groups=2):
    requesters = [make_user(i) for i in range(1, n_tickets + 1)]
    agents = [make_user(1000 + i, role="agent") for i in range(n_assignees)]
    tickets = [
        make_ticket(i, requester_id=i, assignee_id=1000 + (i % n_assignees), group_id=5000 + (i % n_groups))
        for i in range(1, n_tickets + 1)
    ]
    groups = [
        {"id": 5000 + g, "name": f"G{g}", "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z"}
        for g in range(n_groups)
    ]
    comments = {
        1: [
            {"id": c, "author_id": 1 + (c % 4), "body": "hi", "public": True,
             "created_at": "2026-01-01T00:00:00Z", "attachments": []}
            for c in range(1, 31)
        ]
    }
    return FakeZendesk(tickets, requesters + agents, groups, comments)


@pytest.fixture
def fake(monkeypatch):
    fake = _account()
    patch_zendesk(monkeypatch, fake)
    return fake


def test_search_makes_a_constant_number_of_calls_regardless_of_result_size(fake):
    from zendesk_mcp.tools.ticket import _search_tickets_data
    import json

    result = json.loads(_search_tickets_data("broke", None, 50))

    assert len(result) == 50
    assert result[0]["requester"]["name"].startswith("User ")
    assert result[0]["assignee"]["email"].endswith("@example.com")
    # one search page plus one batched user lookup (was 1 + 55 individual lookups)
    assert len(fake.calls) == 2
    assert any("show_many" in p for p in fake.paths())


def test_search_with_more_than_100_distinct_users_batches_in_chunks_of_100(monkeypatch):
    fake = _account(n_tickets=150)
    patch_zendesk(monkeypatch, fake)
    from zendesk_mcp.tools.ticket import _search_tickets_data

    _search_tickets_data("broke", None, 150)

    show_many = [p for p in fake.paths() if "show_many" in p]
    assert len(show_many) == 2  # 155 users -> 100 + 55
    assert all(len(p.split("ids=")[1].split(",")) <= 100 for p in show_many)
    assert not any(p.startswith("/api/v2/users/") and "show_many" not in p for p in fake.paths())


def test_get_ticket_batches_requester_and_assignee(fake):
    from zendesk_mcp.tools.ticket import _get_ticket_data
    import json

    result = json.loads(_get_ticket_data(1))

    assert result["requester"]["name"] == "User 1"
    assert result["group"] == "G1"
    assert len(fake.calls) == 3  # ticket, users (one batch), group


def test_get_comments_looks_up_all_authors_in_one_call(fake):
    from zendesk_mcp.tools.comments import _get_comments_data
    import json

    result = json.loads(_get_comments_data(1))

    assert len(result) == 30
    assert all(c["author"]["name"].startswith("User ") for c in result)
    assert len(fake.calls) == 2  # comments, users (one batch)


def test_ticket_to_gitlab_context_does_not_look_up_users_one_by_one(fake):
    from zendesk_mcp.tools.gitlab_context import _get_gitlab_context

    out = _get_gitlab_context(1)

    assert "User 1" in out
    assert len(fake.calls) == 3  # ticket, comments, users (one batch)


def test_a_failed_batch_falls_back_to_per_user_lookups(fake, monkeypatch):
    """The prefetch is an optimisation only; the tool must still work if it fails."""
    import zendesk_mcp.lookup as lookup
    from zendesk_mcp.tools.ticket import _search_tickets_data
    import json

    original = fake.send

    def fail_show_many(request, **kw):
        if "show_many" in request.url:
            return fake._respond(request, 500, {"error": "boom"})
        return original(request, **kw)

    monkeypatch.setattr(fake, "send", fail_show_many)
    result = json.loads(_search_tickets_data("broke", None, 3))

    assert len(result) == 3
    assert result[0]["requester"]["name"] == "User 1"
    assert lookup.prefetch_users  # imported and in use
