"""Batched lookups that spare the API one request per related record.

Zenpy resolves ``ticket.requester``, ``ticket.assignee`` and a comment's author lazily, one
``GET /users/{id}`` each, and caches the result. A search returning 50 tickets therefore cost
about 55 requests, and every tool call builds a new client, so nothing is cached between calls.
Fetching the users in one ``show_many`` request first fills that cache, and the lazy lookups
that follow are served from it.
"""

import logging

log = logging.getLogger(__name__)

_SHOW_MANY_LIMIT = 100  # Zendesk's show_many endpoint accepts at most 100 ids


def prefetch_users(client, user_ids) -> None:
    """Load the given users into the client's cache with as few requests as possible.

    Best effort: if the batch fails the lazy per-user lookups still work (and surface any
    real error), so a failure here is not raised.
    """
    ids = sorted({i for i in user_ids if isinstance(i, int) and not isinstance(i, bool)})
    for start in range(0, len(ids), _SHOW_MANY_LIMIT):
        try:
            list(client.users(ids=ids[start:start + _SHOW_MANY_LIMIT]))
        except Exception as e:
            log.warning("batched user prefetch failed, falling back to per-user lookups: %s", e)
            return
